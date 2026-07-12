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
_CONTINUOUS_PRIMARY_LAYERS = frozenset(
    {"latent_expression", "pre_dropout_expression"}
)
_PROVENANCE_FIELDS = (
    "source",
    "source_sha256",
    "software",
    "software_version",
    "parameters",
    "seeds",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_EVALUATOR_NAME_TOKENS = (
    "label",
    "marker",
    "pseudotime",
    "ground_truth",
    "cell_type",
    "celltype",
)


def _raw_layers(adata: ad.AnnData) -> Mapping[str, object]:
    """Return layers without asking AnnData to validate malformed test inputs."""

    layers = getattr(adata, "_layers", None)
    return layers if isinstance(layers, Mapping) else adata.layers


def _as_numeric_values(matrix: object, name: str) -> np.ndarray:
    if sparse.issparse(matrix):
        canonical = matrix.tocsr(copy=True)
        canonical.sum_duplicates()
        values = canonical.data
    else:
        values = np.asarray(matrix).reshape(-1)
    if np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error


def _validate_nonnegative_matrix(
    matrix: object,
    name: str,
    *,
    integer: bool,
) -> None:
    values = _as_numeric_values(matrix, name)
    finite_nonnegative = np.isfinite(values).all() and bool((values >= 0).all())
    if integer:
        if not finite_nonnegative or not bool((values == np.floor(values)).all()):
            raise ValueError(f"{name} must contain finite nonnegative integers")
    elif not finite_nonnegative:
        raise ValueError(f"{name} must be finite and nonnegative")


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
        if bool(adata.obs[column].isna().any()):
            raise ValueError(f"required obs column {column} contains missing values")

    for column in REQUIRED_OBS_COLUMNS[:-1]:
        if any(isinstance(value, str) and not value.strip() for value in adata.obs[column]):
            raise ValueError(f"required obs column {column} contains empty values")

    try:
        library_size = np.asarray(adata.obs["library_size"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("library_size must be finite and nonnegative") from error
    if not np.isfinite(library_size).all() or not bool((library_size >= 0).all()):
        raise ValueError("library_size must be finite and nonnegative")


def _require_canonical_json(value: object, name: str) -> None:
    def validate_keys(item: object) -> None:
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ValueError(f"{name} must be canonical JSON with string object keys")
            for nested in item.values():
                validate_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                validate_keys(nested)

    validate_keys(value)
    try:
        json.dumps(
            value,
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
    return lowered in {"group", "groups", "truth"} or any(
        token in lowered for token in _EVALUATOR_NAME_TOKENS
    )


def _parse_allowed_covariates(adata: ad.AnnData) -> tuple[list[str], list[str]]:
    declaration = adata.uns.get("allowed_covariates")
    if declaration is None:
        return [], []
    if isinstance(declaration, Mapping):
        extra = set(declaration) - {"obs", "var"}
        if extra:
            raise ValueError("allowed_covariates may contain only obs and var")
        obs_names = declaration.get("obs", [])
        var_names = declaration.get("var", [])
    else:
        obs_names = declaration
        var_names = []

    def validate_axis(
        names: object, available: pd.Index, axis: str
    ) -> list[str]:
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or not all(isinstance(name, str) and name for name in names)
        ):
            raise ValueError(f"allowed_covariates {axis} must be a list of column names")
        result = list(names)
        if len(result) != len(set(result)):
            raise ValueError(f"allowed_covariates {axis} contains duplicate columns")
        missing = [name for name in result if name not in available]
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
        return result

    return (
        validate_axis(obs_names, adata.obs.columns, "obs"),
        validate_axis(var_names, adata.var.columns, "var"),
    )


def validate_benchmark_dataset(adata: ad.AnnData) -> None:
    """Validate the truth-bearing AnnData publication interchange contract."""

    if not isinstance(adata, ad.AnnData):
        raise TypeError("benchmark dataset must be an AnnData object")
    _validate_ids(adata)
    _validate_obs(adata)
    _validate_nonnegative_matrix(adata.X, "observed counts", integer=True)

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
    return {"type": type(value).__name__, "value": str(value)}


def _frame_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for position, column in enumerate(frame.columns):
        series = frame.iloc[:, position]
        payload.append(
            {
                "name": _normalise_scalar(column),
                "values": [_normalise_scalar(value) for value in series.tolist()],
            }
        )
    return payload


def _matrix_sha256(matrix: object) -> str:
    """Hash matrix semantics, independent of dense/CSR/CSC representation."""

    digest = hashlib.sha256()
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("benchmark matrices must be two-dimensional")
    digest.update(np.asarray(shape, dtype="<u8").tobytes())

    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=True)
        csr.sum_duplicates()
        csr.sort_indices()
        csr.eliminate_zeros()
        for row in range(shape[0]):
            start, stop = csr.indptr[row : row + 2]
            columns = np.asarray(csr.indices[start:stop], dtype="<u8")
            values = np.asarray(csr.data[start:stop], dtype="<f8")
            digest.update(np.asarray([len(columns)], dtype="<u8").tobytes())
            digest.update(columns.tobytes())
            digest.update(values.tobytes())
    else:
        dense = np.asarray(matrix)
        for row in range(shape[0]):
            row_values = np.asarray(dense[row]).reshape(-1)
            columns = np.flatnonzero(row_values != 0).astype("<u8", copy=False)
            values = np.asarray(row_values[columns], dtype="<f8")
            digest.update(np.asarray([len(columns)], dtype="<u8").tobytes())
            digest.update(columns.tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()


def benchmark_dataset_sha256(adata: ad.AnnData) -> str:
    """Hash all observed input, evaluator truth, IDs, and dataset provenance."""

    validate_benchmark_dataset(adata)
    layers = _raw_layers(adata)
    bound_uns = {
        key: deepcopy(adata.uns[key])
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
        "observed_sha256": _matrix_sha256(adata.X),
        "layers": {
            name: _matrix_sha256(layers[name]) for name in sorted(layers)
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


def make_inference_view(adata: ad.AnnData) -> ad.AnnData:
    """Return a deep, truth-free copy containing only declared method inputs."""

    source_sha256 = benchmark_dataset_sha256(adata)
    obs_covariates, var_covariates = _parse_allowed_covariates(adata)
    view = ad.AnnData(
        X=_copy_matrix(adata.X),
        obs=adata.obs.loc[:, obs_covariates].copy(deep=True),
        var=adata.var.loc[:, var_covariates].copy(deep=True),
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
