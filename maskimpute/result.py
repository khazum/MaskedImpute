"""Result containers for MaskImpute inference."""

from collections.abc import Mapping
import copy
from dataclasses import dataclass, FrozenInstanceError
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True, slots=True)
class _ArraySnapshot:
    payload: bytes
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SparseSnapshot:
    data: _ArraySnapshot
    indices: _ArraySnapshot
    indptr: _ArraySnapshot
    shape: tuple[int, int]
    is_array: bool


@dataclass(frozen=True, slots=True)
class _DiagnosticMappingSnapshot:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _DiagnosticSequenceSnapshot:
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _DiagnosticSetSnapshot:
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _DiagnosticScalarSnapshot:
    value: Any


def _snapshot_array(value: np.ndarray) -> _ArraySnapshot:
    contiguous = np.array(value, copy=True, order="C", subok=False)
    return _ArraySnapshot(
        payload=contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        shape=contiguous.shape,
    )


def _materialize_array(snapshot: _ArraySnapshot) -> np.ndarray:
    return np.ndarray(
        snapshot.shape,
        dtype=snapshot.dtype,
        buffer=snapshot.payload,
        order="C",
    )


def _reject_sparse_duplicate_coordinates(value: Any, name: str) -> None:
    coordinates = value.tocoo(copy=True)
    if coordinates.nnz < 2:
        return
    order = np.lexsort((coordinates.col, coordinates.row))
    rows = coordinates.row[order]
    columns = coordinates.col[order]
    if np.any((rows[1:] == rows[:-1]) & (columns[1:] == columns[:-1])):
        raise ValueError(f"{name} must not contain duplicate sparse coordinates")


def _matrix_values(value: Any) -> np.ndarray:
    return value.data if sparse.issparse(value) else value


def _validate_real_array(value: Any, name: str, *, nonnegative: bool) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if value.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    entries = _matrix_values(value)
    if not np.all(np.isfinite(entries)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(entries < 0):
        raise ValueError(f"{name} must be nonnegative")


def _prepare_matrix(value: Any, name: str, *, nonnegative: bool) -> Any:
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} must not be a masked array")
    if sparse.issparse(value):
        if np.ma.isMaskedArray(getattr(value, "data", None)):
            raise TypeError(f"{name} must not contain masked sparse data")
        prepared = value.copy()
        _reject_sparse_duplicate_coordinates(prepared, name)
        prepared = prepared.tocsr(copy=True)
        prepared.sort_indices()
    else:
        prepared = np.array(value, copy=True, order="C", subok=False)
    _validate_real_array(prepared, name, nonnegative=nonnegative)
    return prepared


def _prepare_dense_matrix(value: Any, name: str, *, nonnegative: bool) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} must not be a masked array")
    if sparse.issparse(value):
        raise TypeError(f"{name} must be a dense matrix")
    prepared = np.array(value, copy=True, order="C", subok=False)
    _validate_real_array(prepared, name, nonnegative=nonnegative)
    return prepared


def _snapshot_matrix(value: Any) -> _ArraySnapshot | _SparseSnapshot:
    if not sparse.issparse(value):
        return _snapshot_array(value)
    return _SparseSnapshot(
        data=_snapshot_array(value.data),
        indices=_snapshot_array(value.indices),
        indptr=_snapshot_array(value.indptr),
        shape=tuple(value.shape),
        is_array=isinstance(value, sparse.sparray),
    )


def _materialize_matrix(snapshot: _ArraySnapshot | _SparseSnapshot) -> Any:
    if isinstance(snapshot, _ArraySnapshot):
        return _materialize_array(snapshot)
    constructor = sparse.csr_array if snapshot.is_array else sparse.csr_matrix
    return constructor(
        (
            _materialize_array(snapshot.data),
            _materialize_array(snapshot.indices),
            _materialize_array(snapshot.indptr),
        ),
        shape=snapshot.shape,
        copy=False,
    )


def _snapshot_diagnostic(value: Any, path: str = "diagnostics") -> Any:
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{path} cannot contain masked arrays")
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            items.append((key, _snapshot_diagnostic(item, f"{path}.{key}")))
        return _DiagnosticMappingSnapshot(tuple(items))
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{path} cannot contain object arrays")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{path} must contain only finite values")
        return _snapshot_array(value)
    if isinstance(value, (list, tuple)):
        return _DiagnosticSequenceSnapshot(
            tuple(
                _snapshot_diagnostic(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        )
    if isinstance(value, (set, frozenset)):
        return _DiagnosticSetSnapshot(
            tuple(
                _snapshot_diagnostic(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        )
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be finite")
        return _DiagnosticScalarSnapshot(copy.deepcopy(value))
    if value is None or isinstance(value, (str, bytes, bool)):
        return _DiagnosticScalarSnapshot(copy.deepcopy(value))
    raise TypeError(f"{path} contains an unsupported value")


def _materialize_diagnostic(snapshot: Any) -> Any:
    if isinstance(snapshot, _ArraySnapshot):
        return _materialize_array(snapshot)
    if isinstance(snapshot, _DiagnosticMappingSnapshot):
        return MappingProxyType(
            {key: _materialize_diagnostic(item) for key, item in snapshot.items}
        )
    if isinstance(snapshot, _DiagnosticSequenceSnapshot):
        return tuple(_materialize_diagnostic(item) for item in snapshot.items)
    if isinstance(snapshot, _DiagnosticSetSnapshot):
        return frozenset(_materialize_diagnostic(item) for item in snapshot.items)
    if isinstance(snapshot, _DiagnosticScalarSnapshot):
        return copy.deepcopy(snapshot.value)
    raise TypeError("invalid private diagnostic snapshot")


class ImputationResult:
    """Outputs from a MaskImpute inference run."""

    __slots__ = (
        "_selective_counts",
        "_denoised_counts",
        "_p_pre_zero",
        "_latent",
        "_diagnostics",
    )

    def __init__(
        self,
        selective_counts: Any,
        denoised_counts: Any,
        p_pre_zero: Any,
        latent: Any,
        diagnostics: Any,
    ) -> None:
        selective = _prepare_matrix(
            selective_counts,
            "selective_counts",
            nonnegative=True,
        )
        denoised = _prepare_matrix(
            denoised_counts,
            "denoised_counts",
            nonnegative=True,
        )
        probability = _prepare_dense_matrix(
            p_pre_zero,
            "p_pre_zero",
            nonnegative=True,
        )
        latent_matrix = _prepare_dense_matrix(latent, "latent", nonnegative=False)

        if selective.shape != denoised.shape:
            raise ValueError("selective_counts and denoised_counts shapes must match")
        if probability.shape != selective.shape:
            raise ValueError("p_pre_zero shape must match count matrices")
        if latent_matrix.shape[0] != selective.shape[0]:
            raise ValueError("latent rows must match count-matrix rows")
        if np.any(probability > 1):
            raise ValueError("p_pre_zero probabilities must lie in [0, 1]")

        if not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        diagnostic_snapshot = _snapshot_diagnostic(diagnostics)

        object.__setattr__(self, "_selective_counts", _snapshot_matrix(selective))
        object.__setattr__(self, "_denoised_counts", _snapshot_matrix(denoised))
        object.__setattr__(self, "_p_pre_zero", _snapshot_array(probability))
        object.__setattr__(self, "_latent", _snapshot_array(latent_matrix))
        object.__setattr__(self, "_diagnostics", diagnostic_snapshot)

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    @property
    def selective_counts(self) -> Any:
        return _materialize_matrix(self._selective_counts)

    @property
    def denoised_counts(self) -> Any:
        return _materialize_matrix(self._denoised_counts)

    @property
    def p_pre_zero(self) -> np.ndarray:
        return _materialize_array(self._p_pre_zero)

    @property
    def latent(self) -> np.ndarray:
        return _materialize_array(self._latent)

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return _materialize_diagnostic(self._diagnostics)
