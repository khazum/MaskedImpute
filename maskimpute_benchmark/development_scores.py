"""Canonical development count-score and calibration preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
from typing import Any

import numpy as np

from maskimpute import (
    PreZeroCountModelConfig,
    PreZeroCountModelScore,
    fit_p_pre_zero_count_model,
)
from maskimpute.calibration import (
    CalibrationRecord,
    fit_development_calibration,
    load_calibration_artifact,
    save_calibration_artifact,
)
from maskimpute.count_model import (
    _BUILD_TOKEN,
    _FoldModelRecord,
    _LinkFit,
    _build_score,
    _cell_ids_sha256,
)

from .datasets import validate_dataset_status
from .runner import (
    DEVELOPMENT_MECHANISMS,
    DEVELOPMENT_VIEWS,
    DatasetQCPolicy,
    PreparedDataset,
    prepare_dataset_pair_for_execution,
    validate_development_manifest_payload,
)


_SCORE_MAGIC = b"maskimpute-complete-count-score-v1\0"
_SCORE_ROOT = Path("artifacts/study/development/count_scores")
_CALIBRATION_ROOT = Path("artifacts/study/development/calibration")
_CALIBRATION_NAME = "retained_calibration.json"
_MANIFEST_NAME = "manifest.json"
_ENTRY_FIELDS = {
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
_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "dataset_manifest_sha256",
    "count_model_config_sha256",
    "dataset_qc_policy_sha256",
    "entries",
    "manifest_sha256",
}


class DevelopmentScorePreparationError(RuntimeError):
    """Raised when development score preparation cannot remain canonical."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_owned_regular_bytes(path, "artifact")).hexdigest()


def _read_owned_regular_bytes(path: Path, name: str) -> bytes:
    """Read one owned, single-link regular file without following a symlink."""

    descriptor = -1
    try:
        before_path = path.lstat()
        if (
            not stat.S_ISREG(before_path.st_mode)
            or before_path.st_nlink != 1
            or before_path.st_uid != os.geteuid()
        ):
            raise DevelopmentScorePreparationError(
                f"{name} must be one owned, single-link regular file"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or (opened.st_dev, opened.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise DevelopmentScorePreparationError(
                f"{name} changed while opening or is not uniquely owned"
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        after_path = path.lstat()

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_uid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(opened) != identity(after) or identity(opened) != identity(
            after_path
        ):
            raise DevelopmentScorePreparationError(f"{name} changed while reading")
        return b"".join(chunks)
    except DevelopmentScorePreparationError:
        raise
    except OSError as error:
        raise DevelopmentScorePreparationError(
            f"{name} is unavailable or cannot be read safely"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_owned_directory_inventory(
    directory: Path,
    expected_names: set[str],
    name: str,
) -> None:
    """Validate a directory and every entry through non-following stat calls."""

    descriptor = -1
    try:
        before_path = directory.lstat()
        if not stat.S_ISDIR(before_path.st_mode) or before_path.st_uid != os.geteuid():
            raise DevelopmentScorePreparationError(
                f"existing {name} directory is a symlink or is not owned"
            )
        descriptor = os.open(
            directory,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or (opened.st_dev, opened.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise DevelopmentScorePreparationError(
                f"existing {name} directory changed or is not owned"
            )
        observed_names = set()
        with os.scandir(descriptor) as entries:
            for entry in entries:
                observed_names.add(entry.name)
                metadata = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_uid != os.geteuid()
                ):
                    raise DevelopmentScorePreparationError(
                        f"existing {name} inventory contains a linked or special entry"
                    )
        after_path = directory.lstat()
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_uid) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_uid,
        ) or (after_path.st_dev, after_path.st_ino, after_path.st_uid) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_uid,
        ):
            raise DevelopmentScorePreparationError(
                f"existing {name} directory changed during inventory"
            )
        if observed_names != expected_names:
            raise DevelopmentScorePreparationError(
                f"existing {name} inventory is partial or conflicting"
            )
    except DevelopmentScorePreparationError:
        raise
    except OSError as error:
        raise DevelopmentScorePreparationError(
            f"existing {name} directory cannot be inspected safely"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def canonical_cell_ids_sha256(cell_ids: Sequence[str]) -> str:
    """Use the count-model external-cell identity framing for audit hashes."""

    values = tuple(cell_ids)
    if any(type(value) is not str or not value for value in values):
        raise ValueError("cell IDs must be nonempty exact strings")
    if len(values) != len(set(values)):
        raise ValueError("cell IDs must be unique")
    return _cell_ids_sha256(values)


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _array_bytes(value: object, dtype: str) -> tuple[np.ndarray, bytes]:
    array = np.array(value, dtype=np.dtype(dtype), copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise DevelopmentScorePreparationError("score artifact array is nonfinite")
    return array, array.tobytes(order="C")


def save_count_score_artifact(path: str | Path, score: PreZeroCountModelScore) -> None:
    """Save one complete immutable count score in a deterministic binary format."""

    if type(score) is not PreZeroCountModelScore:
        raise TypeError("score must be an exact PreZeroCountModelScore")
    score_manifest = score.manifest
    arrays: dict[str, tuple[np.ndarray, bytes]] = {}
    for name, value, dtype in (
        ("alpha", score.alpha, "<f8"),
        ("fold_ids", score.fold_ids, "<i8"),
        ("mu", score.mu, "<f8"),
        ("p_pre_zero", score.p_pre_zero, "<f8"),
        ("pi", score.pi, "<f8"),
    ):
        arrays[name] = _array_bytes(value, dtype)
    fold_payload = []
    for fold in score.fold_models:
        means_name = f"fold-{fold.fold_id:03d}-gene-means"
        dispersion_name = f"fold-{fold.fold_id:03d}-gene-dispersion"
        arrays[means_name] = _array_bytes(fold.gene_means, "<f8")
        arrays[dispersion_name] = _array_bytes(fold.gene_dispersion, "<f8")
        fold_payload.append(
            {
                "aggregated_bin_count": fold.aggregated_bin_count,
                "clamp_fraction": fold.clamp_fraction,
                "exposure_reference": fold.exposure_reference,
                "fold_id": fold.fold_id,
                "gene_dispersion_array": dispersion_name,
                "gene_means_array": means_name,
                "held_out_indices": list(fold.held_out_indices),
                "link_converged": fold.link_converged,
                "link_fallback": fold.link_fallback,
                "link_intercept": fold.link_intercept,
                "link_iterations": fold.link_iterations,
                "link_slope": fold.link_slope,
                "training_cell_count": fold.training_cell_count,
                "training_input_sha256": fold.training_input_sha256,
            }
        )
    array_order = tuple(sorted(arrays))
    descriptors = []
    for name in array_order:
        array, raw = arrays[name]
        descriptors.append(
            {
                "dtype": array.dtype.str,
                "name": name,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "shape": list(array.shape),
            }
        )
    unsigned = {
        "schema_version": 1,
        "artifact_type": "maskimpute_complete_count_score",
        "score_manifest": score_manifest,
        "cell_ids": list(score._cell_ids),
        "arrays": descriptors,
        "fold_models": fold_payload,
    }
    metadata = {**unsigned, "metadata_sha256": _canonical_sha256(unsigned)}
    metadata_bytes = _canonical_bytes(metadata)
    output = bytearray(_SCORE_MAGIC)
    output.extend(struct.pack("<Q", len(metadata_bytes)))
    output.extend(metadata_bytes)
    for name in array_order:
        output.extend(arrays[name][1])
    try:
        _publish_bytes(Path(path), bytes(output))
    except (OSError, FileExistsError) as error:
        raise DevelopmentScorePreparationError(
            "count-score artifact could not be published atomically"
        ) from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentScorePreparationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_score_bytes(path: Path) -> bytes:
    return _read_owned_regular_bytes(path, "count-score artifact")


def load_count_score_artifact(path: str | Path) -> PreZeroCountModelScore:
    """Load and semantically reconstitute one complete count-score artifact."""

    raw = _read_score_bytes(Path(path))
    if not raw.startswith(_SCORE_MAGIC) or len(raw) < len(_SCORE_MAGIC) + 8:
        raise DevelopmentScorePreparationError("count-score artifact header is invalid")
    offset = len(_SCORE_MAGIC)
    metadata_size = struct.unpack("<Q", raw[offset : offset + 8])[0]
    offset += 8
    metadata_end = offset + metadata_size
    if metadata_end > len(raw):
        raise DevelopmentScorePreparationError(
            "count-score artifact metadata is truncated"
        )
    try:
        metadata = json.loads(
            raw[offset:metadata_end],
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value}")
            ),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DevelopmentScorePreparationError(
            "count-score artifact metadata is invalid"
        ) from error
    if type(metadata) is not dict or set(metadata) != {
        "schema_version",
        "artifact_type",
        "score_manifest",
        "cell_ids",
        "arrays",
        "fold_models",
        "metadata_sha256",
    }:
        raise DevelopmentScorePreparationError("count-score artifact schema is invalid")
    unsigned = {
        key: value for key, value in metadata.items() if key != "metadata_sha256"
    }
    if (
        metadata["schema_version"] != 1
        or type(metadata["schema_version"]) is not int
        or metadata["artifact_type"] != "maskimpute_complete_count_score"
        or metadata["metadata_sha256"] != _canonical_sha256(unsigned)
        or raw[offset:metadata_end] != _canonical_bytes(metadata)
    ):
        raise DevelopmentScorePreparationError(
            "count-score artifact metadata checksum is invalid"
        )
    offset = metadata_end
    descriptors = metadata["arrays"]
    if type(descriptors) is not list or not descriptors:
        raise DevelopmentScorePreparationError(
            "count-score artifact arrays are invalid"
        )
    arrays: dict[str, np.ndarray] = {}
    observed_names = []
    for descriptor in descriptors:
        if type(descriptor) is not dict or set(descriptor) != {
            "dtype",
            "name",
            "nbytes",
            "sha256",
            "shape",
        }:
            raise DevelopmentScorePreparationError("score array descriptor is invalid")
        name = descriptor["name"]
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        nbytes = descriptor["nbytes"]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(dtype, str)
            or type(shape) is not list
            or any(type(value) is not int or value < 0 for value in shape)
            or type(nbytes) is not int
            or nbytes < 0
        ):
            raise DevelopmentScorePreparationError("score array descriptor is invalid")
        try:
            array_dtype = np.dtype(dtype)
        except TypeError as error:
            raise DevelopmentScorePreparationError(
                "score array dtype is invalid"
            ) from error
        expected_size = int(np.prod(shape, dtype=np.int64)) * array_dtype.itemsize
        if expected_size != nbytes or offset + nbytes > len(raw):
            raise DevelopmentScorePreparationError("score array length is invalid")
        payload = raw[offset : offset + nbytes]
        offset += nbytes
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise DevelopmentScorePreparationError("score array checksum is invalid")
        observed_names.append(name)
        arrays[name] = (
            np.frombuffer(payload, dtype=array_dtype).reshape(tuple(shape)).copy()
        )
    if offset != len(raw) or observed_names != sorted(observed_names):
        raise DevelopmentScorePreparationError(
            "count-score artifact contains trailing or noncanonical arrays"
        )
    manifest = metadata["score_manifest"]
    cell_ids = metadata["cell_ids"]
    if type(manifest) is not dict or type(cell_ids) is not list:
        raise DevelopmentScorePreparationError("score manifest or cell IDs are invalid")
    try:
        config = PreZeroCountModelConfig(**manifest["config"])
        fold_models = []
        folds = metadata["fold_models"]
        if type(folds) is not list:
            raise ValueError("fold_models must be an array")
        for fold in folds:
            if type(fold) is not dict or set(fold) != {
                "aggregated_bin_count",
                "clamp_fraction",
                "exposure_reference",
                "fold_id",
                "gene_dispersion_array",
                "gene_means_array",
                "held_out_indices",
                "link_converged",
                "link_fallback",
                "link_intercept",
                "link_iterations",
                "link_slope",
                "training_cell_count",
                "training_input_sha256",
            }:
                raise ValueError("fold model schema is invalid")
            link = _LinkFit(
                intercept=fold["link_intercept"],
                slope=fold["link_slope"],
                converged=fold["link_converged"],
                fallback=fold["link_fallback"],
                iterations=fold["link_iterations"],
                aggregated_bin_count=fold["aggregated_bin_count"],
            )
            fold_models.append(
                _FoldModelRecord(
                    _BUILD_TOKEN,
                    fold_id=fold["fold_id"],
                    held_out_indices=tuple(fold["held_out_indices"]),
                    training_cell_count=fold["training_cell_count"],
                    training_input_sha256=fold["training_input_sha256"],
                    gene_means=arrays[fold["gene_means_array"]],
                    gene_dispersion=arrays[fold["gene_dispersion_array"]],
                    exposure_reference=fold["exposure_reference"],
                    link=link,
                    clamp_fraction=fold["clamp_fraction"],
                )
            )
        shape = tuple(manifest["shape"])
        score = _build_score(
            counts=np.zeros(shape, dtype=np.float64),
            cell_ids=tuple(cell_ids),
            config=config,
            input_sha256=manifest["input_sha256"],
            fold_ids=arrays["fold_ids"],
            fold_models=tuple(fold_models),
            p_pre_zero=arrays["p_pre_zero"],
            mu=arrays["mu"],
            alpha=arrays["alpha"],
            pi=arrays["pi"],
        )
    except Exception as error:
        raise DevelopmentScorePreparationError(
            "count-score artifact could not be reconstructed"
        ) from error
    if score.manifest != manifest:
        raise DevelopmentScorePreparationError(
            "count-score reconstructed manifest differs from artifact"
        )
    return score


def fit_prepared_count_score(
    prepared: PreparedDataset,
    config: PreZeroCountModelConfig,
) -> PreZeroCountModelScore:
    """Fit a score through the truth-free MethodInput boundary only."""

    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    if type(config) is not PreZeroCountModelConfig:
        raise TypeError("config must be an exact PreZeroCountModelConfig")
    return fit_p_pre_zero_count_model(
        prepared.method_input.counts,
        prepared.method_input.obs_ids,
        config,
    )


def _score_filename(prepared: PreparedDataset) -> str:
    binding = prepared.binding
    return (
        f"{binding.mechanism}--{binding.biological_id}--{binding.technical_view}.score"
    )


def _expected_order() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (mechanism, f"draw-{draw:02d}", view)
        for mechanism in DEVELOPMENT_MECHANISMS
        for draw in (1, 2)
        for view in DEVELOPMENT_VIEWS
    )


def _validate_prepared_panel(
    prepared_datasets: Sequence[PreparedDataset],
) -> tuple[PreparedDataset, ...]:
    values = tuple(prepared_datasets)
    if len(values) != 16 or any(
        not isinstance(value, PreparedDataset) for value in values
    ):
        raise DevelopmentScorePreparationError(
            "development preparation requires exactly 16 prepared datasets"
        )
    observed = tuple(
        (
            value.binding.mechanism,
            value.binding.biological_id,
            value.binding.technical_view,
        )
        for value in values
    )
    if observed != _expected_order():
        raise DevelopmentScorePreparationError(
            "prepared development datasets are not in canonical panel order"
        )
    dataset_ids = set()
    for first, second in zip(values[::2], values[1::2], strict=True):
        if first.audit != second.audit:
            raise DevelopmentScorePreparationError(
                "paired views do not share the exact union cell exclusion"
            )
        for prepared in (first, second):
            binding = prepared.binding
            method_input = prepared.method_input
            if binding.dataset_id in dataset_ids:
                raise DevelopmentScorePreparationError("dataset IDs are duplicated")
            dataset_ids.add(binding.dataset_id)
            if (
                method_input.source_dataset_sha256 != binding.dataset_sha256
                or method_input.shape
                != (prepared.audit.retained_cell_count, binding.genes)
                or method_input.obs_ids != prepared.audit.retained_cell_ids
                or len(method_input.var_ids) != binding.genes
                or prepared.audit.excluded_cell_count
                + prepared.audit.retained_cell_count
                != binding.cells
            ):
                raise DevelopmentScorePreparationError(
                    "prepared dataset violates source, cell, or no-gene-filtering bindings"
                )
        if (
            first.method_input.obs_ids != second.method_input.obs_ids
            or first.method_input.var_ids != second.method_input.var_ids
        ):
            raise DevelopmentScorePreparationError(
                "paired views do not retain identical cells and genes"
            )
    return values


def _score_entry(
    prepared: PreparedDataset,
    score: PreZeroCountModelScore,
) -> dict[str, object]:
    binding = prepared.binding
    audit = prepared.audit
    manifest = score.manifest
    retained_sha = canonical_cell_ids_sha256(audit.retained_cell_ids)
    if manifest["cell_identity"]["digest_sha256"] != retained_sha:
        raise DevelopmentScorePreparationError(
            "score cell identities differ from the retained QC cells"
        )
    return {
        "mechanism": binding.mechanism,
        "biological_id": binding.biological_id,
        "technical_view": binding.technical_view,
        "dataset_id": binding.dataset_id,
        "dataset_sha256": binding.dataset_sha256,
        "input_sha256": manifest["input_sha256"],
        "cell_ids_sha256": retained_sha,
        "excluded_cell_count": audit.excluded_cell_count,
        "excluded_cell_ids_sha256": canonical_cell_ids_sha256(audit.excluded_cell_ids),
        "retained_cell_count": audit.retained_cell_count,
        "retained_cell_ids_sha256": retained_sha,
        "score_sha256": manifest["score_sha256"],
        "config_sha256": manifest["config_sha256"],
    }


def _calibration_record(
    prepared: PreparedDataset,
    score: PreZeroCountModelScore,
) -> CalibrationRecord:
    binding = prepared.binding
    if binding.mechanism != "symsim":
        raise DevelopmentScorePreparationError(
            "only SymSim supplies exact calibration truth"
        )
    dataset = prepared.evaluator_dataset
    if (
        dataset.uns.get("truth_kind") != "exact_pre_capture"
        or dataset.uns.get("primary_truth_layer") != "pre_capture_counts"
        or "pre_capture_counts" not in dataset.layers
    ):
        raise DevelopmentScorePreparationError(
            "SymSim calibration dataset lacks exact pre-capture truth"
        )
    observed = np.asarray(prepared.method_input.counts)
    truth = np.asarray(dataset.layers["pre_capture_counts"])
    if truth.shape != observed.shape:
        raise DevelopmentScorePreparationError(
            "calibration truth and observed counts have different shapes"
        )
    observed_zero = observed == 0
    if not np.any(observed_zero):
        raise DevelopmentScorePreparationError(
            "SymSim calibration record contains no observed zeros"
        )
    probability = score.p_pre_zero[observed_zero]
    target = (truth[observed_zero] == 0).astype(np.int8)
    return CalibrationRecord(
        p_pre_zero=tuple(float(value) for value in probability),
        target=tuple(int(value) for value in target),
        mechanism="symsim",
        biological_id=binding.biological_id,
        manifest_sha256=score.score_sha256,
        truth_kind="exact_pre_capture",
        namespace="dev",
        data_role="development",
        technical_view=binding.technical_view,
        dataset_id=binding.dataset_id,
        dataset_sha256=binding.dataset_sha256,
        protocol_sha256=binding.protocol_sha256,
    )


def _manifest_payload(
    entries: Sequence[Mapping[str, object]],
    *,
    dataset_manifest_sha256: str,
    count_model_config_sha256: str,
    dataset_qc_policy_sha256: str,
) -> dict[str, object]:
    core = {
        "schema_version": 1,
        "artifact_type": "maskimpute_development_count_score_manifest",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "count_model_config_sha256": count_model_config_sha256,
        "dataset_qc_policy_sha256": dataset_qc_policy_sha256,
        "entries": [dict(entry) for entry in entries],
    }
    return {**core, "manifest_sha256": _canonical_sha256(core)}


def _result(
    status: str,
    count_directory: Path,
    calibration_directory: Path,
) -> dict[str, object]:
    manifest_path = count_directory / _MANIFEST_NAME
    calibration_path = calibration_directory / _CALIBRATION_NAME
    manifest = json.loads(manifest_path.read_bytes())
    calibration = load_calibration_artifact(calibration_path)
    payload = calibration.to_dict()
    candidate_metrics = {
        candidate["algorithm"]: {
            "eligible": candidate["eligible"],
            "eligibility_reasons": candidate["eligibility_reasons"],
            "aggregate_metrics": candidate["aggregate_metrics"],
            "biological_draw_metrics": candidate["biological_draw_metrics"],
            "technical_record_metrics": candidate["technical_record_metrics"],
        }
        for candidate in payload["selection"]["candidates"]
    }
    return {
        "status": status,
        "count_score_manifest_path": str(manifest_path),
        "count_score_manifest_payload_sha256": manifest["manifest_sha256"],
        "count_score_manifest_file_sha256": _file_sha256(manifest_path),
        "score_file_sha256s": {
            path.name: _file_sha256(path)
            for path in sorted(count_directory.glob("*.score"))
        },
        "calibration_artifact_path": str(calibration_path),
        "calibration_file_sha256": _file_sha256(calibration_path),
        "calibration_payload_sha256": payload["payload_sha256"],
        "selected_algorithm": calibration.selected_algorithm,
        "calibration_candidates": candidate_metrics,
    }


def _read_canonical_json(path: Path, name: str) -> dict[str, Any]:
    try:
        raw = _read_owned_regular_bytes(path, name)
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value}")
            ),
            object_pairs_hook=_unique_json_object,
        )
    except DevelopmentScorePreparationError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DevelopmentScorePreparationError(f"{name} is invalid") from error
    if type(payload) is not dict or raw != _canonical_bytes(payload) + b"\n":
        raise DevelopmentScorePreparationError(f"{name} is not canonical JSON")
    return payload


def _validate_existing(
    count_directory: Path,
    calibration_directory: Path,
    prepared: tuple[PreparedDataset, ...],
    *,
    dataset_manifest_sha256: str,
    count_model_config: PreZeroCountModelConfig,
    count_model_config_sha256: str,
    dataset_qc_policy_sha256: str,
) -> dict[str, object]:
    expected_score_names = {_score_filename(value) for value in prepared}
    expected_count_files = expected_score_names | {_MANIFEST_NAME}
    _validate_owned_directory_inventory(
        count_directory,
        expected_count_files,
        "count-score output",
    )
    _validate_owned_directory_inventory(
        calibration_directory,
        {_CALIBRATION_NAME},
        "calibration output",
    )
    manifest = _read_canonical_json(
        count_directory / _MANIFEST_NAME,
        "count-score manifest",
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise DevelopmentScorePreparationError(
            "existing count-score manifest schema differs"
        )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest["manifest_sha256"] != _canonical_sha256(unsigned):
        raise DevelopmentScorePreparationError(
            "existing count-score manifest checksum differs"
        )
    entries = manifest["entries"]
    if type(entries) is not list or len(entries) != len(prepared):
        raise DevelopmentScorePreparationError(
            "existing count-score manifest entries are incomplete"
        )
    calibration_records = []
    expected_entries = []
    for prepared_view, existing_entry in zip(prepared, entries, strict=True):
        if type(existing_entry) is not dict or set(existing_entry) != _ENTRY_FIELDS:
            raise DevelopmentScorePreparationError(
                "existing count-score entry schema differs"
            )
        score = load_count_score_artifact(
            count_directory / _score_filename(prepared_view)
        )
        if score.manifest["config"] != asdict(count_model_config):
            raise DevelopmentScorePreparationError(
                "existing score configuration conflicts with tracked authority"
            )
        try:
            score.score_for_counts(
                prepared_view.method_input.counts,
                prepared_view.method_input.obs_ids,
            )
        except Exception as error:
            raise DevelopmentScorePreparationError(
                "existing count-score artifact derivation is invalid"
            ) from error
        expected_entry = _score_entry(prepared_view, score)
        expected_entries.append(expected_entry)
        if existing_entry != expected_entry:
            raise DevelopmentScorePreparationError(
                "existing count-score entry conflicts with its score or dataset"
            )
        if prepared_view.binding.mechanism == "symsim":
            calibration_records.append(_calibration_record(prepared_view, score))
    expected_manifest = _manifest_payload(
        expected_entries,
        dataset_manifest_sha256=dataset_manifest_sha256,
        count_model_config_sha256=count_model_config_sha256,
        dataset_qc_policy_sha256=dataset_qc_policy_sha256,
    )
    if manifest != expected_manifest:
        raise DevelopmentScorePreparationError(
            "existing count-score manifest conflicts with current authority"
        )
    calibration = load_calibration_artifact(calibration_directory / _CALIBRATION_NAME)
    expected_calibration = fit_development_calibration(calibration_records)
    if calibration.to_dict() != expected_calibration.to_dict():
        raise DevelopmentScorePreparationError(
            "existing calibration artifact conflicts with exact score/truth records"
        )
    if any(
        entry["config_sha256"] != count_model_config_sha256
        for entry in expected_entries
    ):
        raise DevelopmentScorePreparationError(
            "existing score configuration conflicts with tracked authority"
        )
    return _result("reused", count_directory, calibration_directory)


def prepare_validated_development_scores(
    repository: str | Path,
    *,
    prepared_datasets: Sequence[PreparedDataset],
    dataset_manifest_sha256: str,
    count_model_config: PreZeroCountModelConfig,
    count_model_config_sha256: str,
    dataset_qc_policy_sha256: str,
) -> dict[str, object]:
    """Prepare all artifacts after dataset bytes and pair QC have been validated."""

    repo = Path(repository).resolve()
    prepared = _validate_prepared_panel(prepared_datasets)
    if type(count_model_config) is not PreZeroCountModelConfig:
        raise TypeError("count_model_config must be exact PreZeroCountModelConfig")
    if _canonical_sha256(asdict(count_model_config)) != count_model_config_sha256:
        raise DevelopmentScorePreparationError(
            "count-model configuration checksum differs from configuration"
        )
    for value, name in (
        (dataset_manifest_sha256, "dataset manifest checksum"),
        (count_model_config_sha256, "count-model configuration checksum"),
        (dataset_qc_policy_sha256, "dataset QC policy checksum"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise DevelopmentScorePreparationError(f"{name} is invalid")
    count_directory = repo / _SCORE_ROOT
    calibration_directory = repo / _CALIBRATION_ROOT
    count_exists = count_directory.exists() or count_directory.is_symlink()
    calibration_exists = (
        calibration_directory.exists() or calibration_directory.is_symlink()
    )
    if count_exists or calibration_exists:
        if not count_exists or not calibration_exists:
            raise DevelopmentScorePreparationError(
                "partial existing score/calibration output fails closed"
            )
        try:
            return _validate_existing(
                count_directory,
                calibration_directory,
                prepared,
                dataset_manifest_sha256=dataset_manifest_sha256,
                count_model_config=count_model_config,
                count_model_config_sha256=count_model_config_sha256,
                dataset_qc_policy_sha256=dataset_qc_policy_sha256,
            )
        except Exception as error:
            raise DevelopmentScorePreparationError(
                "existing score/calibration output failed closed validation"
            ) from error

    development_root = repo / "artifacts/study/development"
    development_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-scores-", dir=development_root))
    stage_count = staging / "count_scores"
    stage_calibration = staging / "calibration"
    stage_count.mkdir(mode=0o700)
    stage_calibration.mkdir(mode=0o700)
    try:
        entries = []
        calibration_records = []
        score_hashes = set()
        for prepared_view in prepared:
            score = fit_prepared_count_score(prepared_view, count_model_config)
            if score.score_sha256 in score_hashes:
                raise DevelopmentScorePreparationError(
                    "count-score hashes are unexpectedly duplicated"
                )
            score_hashes.add(score.score_sha256)
            save_count_score_artifact(
                stage_count / _score_filename(prepared_view),
                score,
            )
            entries.append(_score_entry(prepared_view, score))
            if prepared_view.binding.mechanism == "symsim":
                calibration_records.append(_calibration_record(prepared_view, score))
        manifest = _manifest_payload(
            entries,
            dataset_manifest_sha256=dataset_manifest_sha256,
            count_model_config_sha256=count_model_config_sha256,
            dataset_qc_policy_sha256=dataset_qc_policy_sha256,
        )
        _publish_bytes(
            stage_count / _MANIFEST_NAME,
            _canonical_bytes(manifest) + b"\n",
        )
        calibration = fit_development_calibration(calibration_records)
        save_calibration_artifact(
            stage_calibration / _CALIBRATION_NAME,
            calibration,
        )
        _validate_existing(
            stage_count,
            stage_calibration,
            prepared,
            dataset_manifest_sha256=dataset_manifest_sha256,
            count_model_config=count_model_config,
            count_model_config_sha256=count_model_config_sha256,
            dataset_qc_policy_sha256=dataset_qc_policy_sha256,
        )
        os.rename(stage_count, count_directory)
        try:
            os.rename(stage_calibration, calibration_directory)
        except Exception:
            shutil.rmtree(count_directory)
            raise
        directory = os.open(
            development_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception as error:
        if isinstance(error, DevelopmentScorePreparationError):
            raise
        raise DevelopmentScorePreparationError(
            "development score preparation failed before complete publication"
        ) from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return _result("created", count_directory, calibration_directory)


def _load_selection_contract(
    repository: Path,
) -> tuple[
    PreZeroCountModelConfig,
    str,
    str,
]:
    path = repository / "study/selection_contract.json"
    try:
        contract = json.loads(
            path.read_bytes(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value}")
            ),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DevelopmentScorePreparationError(
            "tracked selection contract is invalid"
        ) from error
    if type(contract) is not dict:
        raise DevelopmentScorePreparationError("selection contract must be an object")
    try:
        config_payload = contract["count_model_config"]
        config_sha = contract["count_model_config_sha256"]
        policy_payload = contract["dataset_qc_policy"]
        policy_sha = contract["dataset_qc_policy_sha256"]
        config = PreZeroCountModelConfig(**config_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise DevelopmentScorePreparationError(
            "selection contract score authority is invalid"
        ) from error
    policy = DatasetQCPolicy.fixed()
    if (
        config_sha != _canonical_sha256(config_payload)
        or asdict(config) != config_payload
        or policy_payload != policy.to_dict()
        or policy_sha != _canonical_sha256(policy_payload)
    ):
        raise DevelopmentScorePreparationError(
            "selection contract score or QC checksum differs"
        )
    return config, config_sha, policy_sha


def prepare_development_scores(repository: str | Path) -> dict[str, object]:
    """Validate the real 16-view panel and prepare its canonical score authority."""

    import anndata as ad

    repo = Path(repository).resolve(strict=True)
    config, config_sha, policy_sha = _load_selection_contract(repo)
    status_path = repo / "artifacts/study/development/results/dataset_status.json"
    try:
        status = validate_dataset_status(status_path, repo=repo)
        bindings = validate_development_manifest_payload(status)
    except Exception as error:
        raise DevelopmentScorePreparationError(
            "completed development dataset status failed byte revalidation"
        ) from error
    prepared = []
    policy = DatasetQCPolicy.fixed()
    for first_binding, second_binding in zip(
        bindings[::2], bindings[1::2], strict=True
    ):
        first_path = (
            repo / "artifacts/study/development/results" / first_binding.output_path
        )
        second_path = (
            repo / "artifacts/study/development/results" / second_binding.output_path
        )
        try:
            first_dataset = ad.read_h5ad(first_path)
            second_dataset = ad.read_h5ad(second_path)
            first, second = prepare_dataset_pair_for_execution(
                first_dataset,
                second_dataset,
                first_binding,
                second_binding,
                policy,
            )
        except Exception as error:
            raise DevelopmentScorePreparationError(
                "development dataset pair failed union-QC preparation"
            ) from error
        prepared.extend((first, second))
    return prepare_validated_development_scores(
        repo,
        prepared_datasets=tuple(prepared),
        dataset_manifest_sha256=status["manifest_sha256"],
        count_model_config=config,
        count_model_config_sha256=config_sha,
        dataset_qc_policy_sha256=policy_sha,
    )


__all__ = [
    "DevelopmentScorePreparationError",
    "canonical_cell_ids_sha256",
    "fit_prepared_count_score",
    "load_count_score_artifact",
    "prepare_development_scores",
    "prepare_validated_development_scores",
    "save_count_score_artifact",
]
