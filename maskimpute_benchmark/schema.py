"""Validated benchmark datasets and leakage-resistant inference views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .protocol import canonical_sha256


class TruthKind(str, Enum):
    """Strength and estimand of the evaluator-only reference matrix."""

    EXACT_PRE_CAPTURE = "exact_pre_capture"
    EXACT_CONTINUOUS = "exact_continuous"
    PROXY_HIGH_DEPTH = "proxy_high_depth"
    ORTHOGONAL_ONLY = "orthogonal_only"

    # Lower-case aliases make the enum mirror the serialized contract.
    exact_pre_capture = EXACT_PRE_CAPTURE
    exact_continuous = EXACT_CONTINUOUS
    proxy_high_depth = PROXY_HIGH_DEPTH
    orthogonal_only = ORTHOGONAL_ONLY


REQUIRED_OBS_COLUMNS = (
    "dataset_id",
    "mechanism",
    "condition",
    "biological_id",
    "technical_view",
    "draw",
    "library_size",
)

EVALUATOR_LAYERS = frozenset(
    {
        "pre_capture_counts",
        "latent_expression",
        "pre_dropout_expression",
        "reference_counts",
        "heldout_counts",
        "expected_counts",
    }
)

_DISCRETE_LAYERS = frozenset(
    {"pre_capture_counts", "reference_counts", "heldout_counts"}
)
_CONTINUOUS_PRIMARY_LAYERS = frozenset({"latent_expression", "pre_dropout_expression"})
_PROVENANCE_FIELDS = (
    "source",
    "source_sha256",
    "software",
    "software_version",
    "parameters",
    "seeds",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_UNS_KEYS = frozenset(
    {
        "truth_kind",
        "primary_truth_layer",
        "provenance",
        "normalization",
        "allowed_covariates",
    }
)
_NORMALIZATION_KEYS = frozenset({"input", "target_sum", "log_base", "size_factor"})
_EVALUATOR_NAME_TOKENS = frozenset(
    {
        "biological",
        "branch",
        "class",
        "classes",
        "classification",
        "cluster",
        "clusters",
        "condition",
        "dataset",
        "disease",
        "draw",
        "group",
        "groups",
        "label",
        "labels",
        "library",
        "lineage",
        "marker",
        "markers",
        "mechanism",
        "outcome",
        "phenotype",
        "pseudotime",
        "response",
        "state",
        "status",
        "technical",
        "timepoint",
        "trajectory",
        "treatment",
        "truth",
    }
)
_EVALUATOR_EXACT_NAMES = frozenset(
    {"cell_type", "celltype", "ground_truth", "case_control", "casecontrol"}
)
_EVALUATOR_TOKEN_PAIRS = frozenset({("cell", "type"), ("case", "control")})


def _raw_layers(adata: ad.AnnData) -> Mapping[str, object]:
    """Return layers without asking AnnData to validate malformed test inputs."""

    layers = getattr(adata, "_layers", None)
    return layers if isinstance(layers, Mapping) else adata.layers


def _native_matrix_values(matrix: object, name: str) -> np.ndarray:
    if sparse.issparse(matrix):
        canonical = matrix.tocsr(copy=True)
        canonical.sum_duplicates()
        values = canonical.data
    else:
        values = np.asarray(matrix).reshape(-1)
    if values.dtype.kind not in {"b", "i", "u", "f"} or values.dtype.itemsize > 8:
        raise ValueError(
            f"{name} must use a native bool/integer/float dtype no wider than float64"
        )
    return values


def _integer_is_exact_float64(value: object) -> bool:
    integer = int(value)
    return int(float(integer)) == integer


def _validate_nonnegative_matrix(
    matrix: object,
    name: str,
    *,
    integer: bool,
) -> None:
    values = _native_matrix_values(matrix, name)
    finite_nonnegative = np.isfinite(values).all() and bool((values >= 0).all())
    if integer:
        if not finite_nonnegative or (
            values.dtype.kind == "f" and not bool((values == np.floor(values)).all())
        ):
            raise ValueError(f"{name} must contain finite nonnegative integers")
        if values.dtype.kind == "f" and bool(
            (values.astype(np.float64, copy=False) >= float(2**64)).any()
        ):
            raise ValueError(f"{name} integer values must fit uint64")
    elif not finite_nonnegative:
        raise ValueError(f"{name} must be finite and nonnegative")
    elif values.dtype.kind in {"b", "i", "u"}:
        large = values[values > 2**53]
        if any(not _integer_is_exact_float64(value) for value in large):
            raise ValueError(
                f"{name} integer values must be exactly representable as float64"
            )


def _observed_row_sums(matrix: object) -> list[int]:
    """Sum validated integer counts without a float precision round trip."""

    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=True)
        csr.sum_duplicates()
        return [
            sum(int(value) for value in csr.data[csr.indptr[row] : csr.indptr[row + 1]])
            for row in range(csr.shape[0])
        ]
    dense = np.asarray(matrix)
    return [sum(int(value) for value in dense[row]) for row in range(dense.shape[0])]


def _validate_ids(adata: ad.AnnData) -> None:
    for axis, names in (("obs", adata.obs_names), ("var", adata.var_names)):
        if not names.is_unique:
            raise ValueError(f"{axis} IDs must be unique")
        if any(not isinstance(value, str) or not value.strip() for value in names):
            raise ValueError(f"{axis} IDs must be nonempty strings")


def _validate_obs(adata: ad.AnnData) -> None:
    if not adata.obs.columns.is_unique:
        raise ValueError("obs column names must be unique")
    if not adata.var.columns.is_unique:
        raise ValueError("var column names must be unique")

    for column in REQUIRED_OBS_COLUMNS:
        if column not in adata.obs:
            raise ValueError(f"missing required obs column: {column}")

    design_columns = (
        "dataset_id",
        "mechanism",
        "condition",
        "biological_id",
        "technical_view",
    )
    for column in design_columns:
        values = adata.obs[column].tolist()
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{column} must contain nonempty strings")
        if len(set(values)) != 1:
            raise ValueError(f"{column} must be constant within an AnnData dataset")

    draw_values = adata.obs["draw"].tolist()
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in draw_values
    ):
        raise ValueError("draw must contain a positive integer")
    if len({int(value) for value in draw_values}) != 1:
        raise ValueError("draw must be constant within an AnnData dataset")

    library_values = adata.obs["library_size"].tolist()
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in library_values
    ):
        raise ValueError("library_size must contain finite nonnegative integers")
    if [int(value) for value in library_values] != _observed_row_sums(adata.X):
        raise ValueError("library_size must exactly equal observed-count row sums")


def _canonical_json_value(
    value: object,
    name: str,
    _active: set[int] | None = None,
) -> object:
    """Normalize AnnData's NumPy-backed JSON scalars without weakening JSON."""

    if _active is None:
        _active = set()
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must be canonical JSON")
    if isinstance(value, np.ndarray):
        if value.dtype.metadata is not None:
            raise ValueError(f"{name} must be canonical JSON")
        identity = id(value)
        if identity in _active:
            raise ValueError(f"{name} must be canonical JSON")
        _active.add(identity)
        try:
            return _canonical_json_value(value.tolist(), name, _active)
        finally:
            _active.remove(identity)
    if isinstance(value, np.generic):
        if value.dtype.metadata is not None:
            raise ValueError(f"{name} must be canonical JSON")
        return _canonical_json_value(value.item(), name, _active)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{name} must be canonical JSON with string object keys")
        identity = id(value)
        if identity in _active:
            raise ValueError(f"{name} must be canonical JSON")
        _active.add(identity)
        try:
            return {
                key: _canonical_json_value(nested, name, _active)
                for key, nested in value.items()
            }
        finally:
            _active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in _active:
            raise ValueError(f"{name} must be canonical JSON")
        _active.add(identity)
        try:
            return [_canonical_json_value(nested, name, _active) for nested in value]
        finally:
            _active.remove(identity)
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{name} must be canonical JSON")


def _require_canonical_json(value: object, name: str) -> None:
    normalized = _canonical_json_value(value, name)
    try:
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be canonical JSON") from error


def _validate_provenance(adata: ad.AnnData) -> None:
    provenance = adata.uns.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a JSON object")
    for field in _PROVENANCE_FIELDS:
        if field not in provenance:
            raise ValueError(f"provenance is missing required field {field}")
    for field in ("source", "software", "software_version"):
        value = provenance[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"provenance {field} must be a nonempty string")
    source_sha256 = provenance["source_sha256"]
    if not isinstance(source_sha256, str) or not _SHA256_PATTERN.fullmatch(
        source_sha256
    ):
        raise ValueError("provenance source_sha256 must be a lowercase SHA-256 digest")
    for field in ("parameters", "seeds"):
        if not isinstance(provenance[field], Mapping):
            raise ValueError(f"provenance {field} must be a JSON object")
    _require_canonical_json(provenance, "provenance canonical JSON")


def _validate_normalization(adata: ad.AnnData) -> None:
    if "normalization" not in adata.uns:
        return
    normalization = adata.uns["normalization"]
    if not isinstance(normalization, Mapping):
        raise ValueError("normalization must be an object with whitelisted scalar keys")
    unsupported = set(normalization) - _NORMALIZATION_KEYS
    if unsupported:
        names = ", ".join(sorted(str(key) for key in unsupported))
        raise ValueError(f"normalization contains unsupported keys: {names}")
    if "input" not in normalization:
        raise ValueError("normalization must declare its input")
    input_value = normalization["input"]
    if not isinstance(input_value, str) or not input_value.strip():
        raise ValueError("normalization input must be a nonempty string")

    for key in ("target_sum", "log_base"):
        if key not in normalization or normalization[key] is None:
            continue
        value = normalization[key]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or (key == "log_base" and value == 1)
        ):
            raise ValueError(f"normalization {key} must be a valid positive number")

    if "size_factor" in normalization and normalization["size_factor"] is not None:
        value = normalization["size_factor"]
        valid_name = isinstance(value, str) and bool(value.strip())
        valid_number = (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
        )
        if not (valid_name or valid_number):
            raise ValueError(
                "normalization size_factor must be a nonempty name or positive number"
            )


def _validate_closed_slots(adata: ad.AnnData) -> None:
    for slot in ("obsm", "varm", "obsp", "varp"):
        if len(getattr(adata, slot)):
            raise ValueError(f"unsupported AnnData slot must be empty: {slot}")
    if adata.raw is not None:
        raise ValueError("unsupported AnnData slot must be empty: raw")
    unsupported_uns = set(adata.uns) - _ALLOWED_UNS_KEYS
    if unsupported_uns:
        names = ", ".join(sorted(str(key) for key in unsupported_uns))
        raise ValueError(f"unsupported uns key: {names}")


def _parse_truth_kind(adata: ad.AnnData) -> TruthKind:
    try:
        return TruthKind(adata.uns.get("truth_kind"))
    except (TypeError, ValueError) as error:
        choices = ", ".join(kind.value for kind in TruthKind)
        raise ValueError(f"truth_kind must be one of: {choices}") from error


def _validate_truth_contract(adata: ad.AnnData, kind: TruthKind) -> None:
    layers = _raw_layers(adata)
    has_primary = "primary_truth_layer" in adata.uns
    primary = adata.uns.get("primary_truth_layer")

    if kind is TruthKind.ORTHOGONAL_ONLY:
        if has_primary:
            raise ValueError("orthogonal_only datasets must not declare primary truth")
        truth_layers = sorted(EVALUATOR_LAYERS & set(layers))
        if truth_layers:
            raise ValueError(
                "orthogonal_only datasets must not contain evaluator truth layers: "
                + ", ".join(truth_layers)
            )
        return

    if not isinstance(primary, str) or not primary:
        raise ValueError("primary_truth_layer must name an evaluator layer")
    if kind is TruthKind.EXACT_PRE_CAPTURE and primary != "pre_capture_counts":
        raise ValueError("exact_pre_capture primary truth must be pre_capture_counts")
    if kind is TruthKind.EXACT_CONTINUOUS and primary not in _CONTINUOUS_PRIMARY_LAYERS:
        raise ValueError(
            "exact_continuous primary must be a continuous truth layer: "
            "latent_expression or pre_dropout_expression"
        )
    if kind is TruthKind.PROXY_HIGH_DEPTH and primary != "reference_counts":
        raise ValueError("proxy_high_depth primary truth must be reference_counts")
    if primary not in layers:
        raise ValueError(f"primary truth layer {primary} is missing")


def _is_evaluator_metadata(name: str) -> bool:
    lowered = name.casefold()
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    token_list = re.findall(r"[a-z]+|[0-9]+", camel_split.casefold())
    tokens = frozenset(token_list)
    token_pairs = frozenset(zip(token_list, token_list[1:]))
    return (
        lowered in REQUIRED_OBS_COLUMNS
        or lowered in _EVALUATOR_EXACT_NAMES
        or bool(tokens & _EVALUATOR_NAME_TOKENS)
        or bool(tokens & _EVALUATOR_EXACT_NAMES)
        or bool(token_pairs & _EVALUATOR_TOKEN_PAIRS)
    )


def _validate_covariate_series(series: pd.Series, axis: str, name: str) -> None:
    label = f"allowed_covariates {axis}.{name}"
    if isinstance(series.dtype, pd.CategoricalDtype):
        if bool(series.isna().any()):
            raise ValueError(f"{label} must contain immutable scalar values")
        categories = series.cat.categories.tolist()
        if any(
            isinstance(value, np.generic)
            or isinstance(value, (bytes, bytearray))
            or not isinstance(value, (str, bool, int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in categories
        ):
            raise ValueError(f"{label} categories must be immutable scalar values")
        return

    if pd.api.types.is_object_dtype(series.dtype):
        if bool(series.isna().any()) or not all(
            isinstance(value, str) for value in series.tolist()
        ):
            raise ValueError(f"{label} must contain immutable scalar strings")
        return
    if isinstance(series.dtype, pd.StringDtype):
        if bool(series.isna().any()):
            raise ValueError(f"{label} must contain immutable scalar strings")
        return
    if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_integer_dtype(
        series.dtype
    ):
        if bool(series.isna().any()):
            raise ValueError(f"{label} must contain immutable scalar values")
        return
    if pd.api.types.is_float_dtype(series.dtype):
        try:
            values = np.asarray(series, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must contain immutable scalar values") from error
        if np.isfinite(values).all():
            return
    raise ValueError(f"{label} must use a supported immutable scalar covariate dtype")


def _parse_allowed_covariates(adata: ad.AnnData) -> tuple[list[str], list[str]]:
    declaration = adata.uns.get("allowed_covariates")
    if declaration is None:
        return [], []
    declaration = _canonical_json_value(
        declaration, "allowed_covariates canonical JSON"
    )
    if isinstance(declaration, Mapping):
        extra = set(declaration) - {"obs", "var"}
        if extra:
            raise ValueError("allowed_covariates may contain only obs and var")
        obs_names = declaration.get("obs", [])
        var_names = declaration.get("var", [])
    else:
        obs_names = declaration
        var_names = []

    def validate_axis(names: object, frame: pd.DataFrame, axis: str) -> list[str]:
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or not all(isinstance(name, str) and name for name in names)
        ):
            raise ValueError(
                f"allowed_covariates {axis} must be a list of column names"
            )
        result = list(names)
        if len(result) != len(set(result)):
            raise ValueError(f"allowed_covariates {axis} contains duplicate columns")
        missing = [name for name in result if name not in frame.columns]
        if missing:
            raise ValueError(
                f"allowed_covariates {axis} columns are missing: {', '.join(missing)}"
            )
        forbidden = [name for name in result if _is_evaluator_metadata(name)]
        if forbidden:
            raise ValueError(
                "evaluator metadata cannot be an allowed covariate: "
                + ", ".join(forbidden)
            )
        for name in result:
            _validate_covariate_series(frame[name], axis, name)
        return result

    return (
        validate_axis(obs_names, adata.obs, "obs"),
        validate_axis(var_names, adata.var, "var"),
    )


def validate_benchmark_dataset(adata: ad.AnnData) -> None:
    """Validate the truth-bearing AnnData publication interchange contract."""

    if not isinstance(adata, ad.AnnData):
        raise TypeError("benchmark dataset must be an AnnData object")
    _validate_closed_slots(adata)
    _validate_ids(adata)
    _validate_nonnegative_matrix(adata.X, "observed counts", integer=True)
    _validate_obs(adata)

    layers = _raw_layers(adata)
    for name, matrix in layers.items():
        if getattr(matrix, "shape", None) != adata.shape:
            raise ValueError(
                f"evaluator layer {name} has shape {getattr(matrix, 'shape', None)}; "
                f"expected {adata.shape}"
            )
        _validate_nonnegative_matrix(
            matrix,
            f"layer {name}",
            integer=name in _DISCRETE_LAYERS,
        )

    truth_kind = _parse_truth_kind(adata)
    _validate_truth_contract(adata, truth_kind)
    _validate_provenance(adata)
    _validate_normalization(adata)
    _parse_allowed_covariates(adata)


def _normalise_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return {"type": "missing"}
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return {"type": "missing"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata values must be finite")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, Mapping):
        entries = [
            (_normalise_scalar(key), _normalise_scalar(nested))
            for key, nested in value.items()
        ]
        entries.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return {"type": "mapping", "entries": entries}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "values": [_normalise_scalar(nested) for nested in value],
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": [_normalise_scalar(nested) for nested in value.reshape(-1)],
        }
    raise ValueError(f"unsupported metadata scalar type: {type(value).__name__}")


def _series_schema(series: pd.Series) -> dict[str, object]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return {
            "kind": "categorical",
            "categories": [
                _normalise_scalar(value) for value in series.cat.categories.tolist()
            ],
            "ordered": bool(series.cat.ordered),
        }
    return {"kind": "plain", "dtype": str(series.dtype)}


def _frame_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for position, column in enumerate(frame.columns):
        series = frame.iloc[:, position]
        payload.append(
            {
                "name": _normalise_scalar(column),
                "schema": _series_schema(series),
                "values": [_normalise_scalar(value) for value in series.tolist()],
            }
        )
    return payload


def _matrix_sha256(matrix: object, *, discrete: bool) -> str:
    """Hash matrix semantics, independent of dense/CSR/CSC representation."""

    digest = hashlib.sha256()
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("benchmark matrices must be two-dimensional")
    digest.update(np.asarray(shape, dtype="<u8").tobytes())
    value_dtype = np.dtype("<u8") if discrete else np.dtype("<f8")
    digest.update(b"discrete-uint64" if discrete else b"continuous-float64")

    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=True)
        csr.sum_duplicates()
        csr.sort_indices()
        csr.eliminate_zeros()
        for row in range(shape[0]):
            start, stop = csr.indptr[row : row + 2]
            columns = np.asarray(csr.indices[start:stop], dtype="<u8")
            values = np.asarray(csr.data[start:stop], dtype=value_dtype)
            digest.update(np.asarray([len(columns)], dtype="<u8").tobytes())
            digest.update(columns.tobytes())
            digest.update(values.tobytes())
    else:
        dense = np.asarray(matrix)
        for row in range(shape[0]):
            row_values = np.asarray(dense[row]).reshape(-1)
            columns = np.flatnonzero(row_values != 0).astype("<u8", copy=False)
            values = np.asarray(row_values[columns], dtype=value_dtype)
            digest.update(np.asarray([len(columns)], dtype="<u8").tobytes())
            digest.update(columns.tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()


def benchmark_dataset_sha256(adata: ad.AnnData) -> str:
    """Hash all observed input, evaluator truth, IDs, and dataset provenance."""

    validate_benchmark_dataset(adata)
    layers = _raw_layers(adata)
    bound_uns = {
        key: _canonical_json_value(
            deepcopy(adata.uns[key]), f"uns.{key} canonical JSON"
        )
        for key in (
            "truth_kind",
            "primary_truth_layer",
            "provenance",
            "normalization",
            "allowed_covariates",
        )
        if key in adata.uns
    }
    payload: dict[str, Any] = {
        "schema": "maskimpute-benchmark-dataset-v1",
        "shape": list(adata.shape),
        "observed_sha256": _matrix_sha256(adata.X, discrete=True),
        "layers": {
            name: _matrix_sha256(layers[name], discrete=name in _DISCRETE_LAYERS)
            for name in sorted(layers)
        },
        "obs_ids": [_normalise_scalar(value) for value in adata.obs_names],
        "var_ids": [_normalise_scalar(value) for value in adata.var_names],
        "obs_metadata": _frame_payload(adata.obs),
        "var_metadata": _frame_payload(adata.var),
        "uns": bound_uns,
    }
    return canonical_sha256(payload)


def _copy_matrix(matrix: object) -> object:
    if sparse.issparse(matrix):
        return matrix.copy()
    return np.array(matrix, copy=True)


def _copy_covariate_frame(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    copied = pd.DataFrame(index=frame.index.copy(deep=True))
    for name in names:
        series = frame[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = [deepcopy(value) for value in series.cat.categories.tolist()]
            categorical = pd.Categorical.from_codes(
                series.cat.codes.to_numpy(copy=True),
                categories=categories,
                ordered=series.cat.ordered,
            )
            copied[name] = pd.Series(categorical, index=copied.index, name=name)
        else:
            copied[name] = series.copy(deep=True)
    return copied


def make_inference_view(adata: ad.AnnData) -> ad.AnnData:
    """Return a deep, truth-free copy containing only declared method inputs."""

    source_sha256 = benchmark_dataset_sha256(adata)
    obs_covariates, var_covariates = _parse_allowed_covariates(adata)
    view = ad.AnnData(
        X=_copy_matrix(adata.X),
        obs=_copy_covariate_frame(adata.obs, obs_covariates),
        var=_copy_covariate_frame(adata.var, var_covariates),
    )
    if "normalization" in adata.uns:
        view.uns["normalization"] = deepcopy(adata.uns["normalization"])
    if "allowed_covariates" in adata.uns:
        view.uns["allowed_covariates"] = deepcopy(adata.uns["allowed_covariates"])
    view.uns["source_dataset_sha256"] = source_sha256
    return view


__all__ = [
    "TruthKind",
    "benchmark_dataset_sha256",
    "make_inference_view",
    "validate_benchmark_dataset",
]
