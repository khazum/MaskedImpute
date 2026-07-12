"""Trusted snapshot boundary for exact SciPy sparse containers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import sparse


SUPPORTED_SPARSE_TYPES = tuple(
    constructor
    for name in (
        "bsr_matrix",
        "coo_matrix",
        "csc_matrix",
        "csr_matrix",
        "dia_matrix",
        "dok_matrix",
        "lil_matrix",
        "bsr_array",
        "coo_array",
        "csc_array",
        "csr_array",
        "dia_array",
        "dok_array",
        "lil_array",
    )
    if isinstance((constructor := getattr(sparse, name, None)), type)
)
_SPARSE_FORMAT_BY_TYPE = {
    constructor: format_name
    for format_name in ("bsr", "coo", "csc", "csr", "dia", "dok", "lil")
    for suffix in ("matrix", "array")
    if isinstance(
        (constructor := getattr(sparse, f"{format_name}_{suffix}", None)),
        type,
    )
}
_EXACT_INTEGER_SCALAR_TYPES = {
    int,
    *(
        scalar_type
        for name in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        )
        if isinstance((scalar_type := getattr(np, name, None)), type)
    ),
}
_EXACT_NUMERIC_SCALAR_TYPES = {
    bool,
    complex,
    float,
    *_EXACT_INTEGER_SCALAR_TYPES,
    *(
        scalar_type
        for name in (
            "bool_",
            "float16",
            "float32",
            "float64",
            "longdouble",
            "complex64",
            "complex128",
            "clongdouble",
        )
        if isinstance((scalar_type := getattr(np, name, None)), type)
    ),
}


def contains_masked_array(value: object, seen: set[int] | None = None) -> bool:
    """Find masked values before coercion can erase their semantics."""

    if np.ma.isMaskedArray(value):
        return True
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if sparse.issparse(value):
        containers = []
        if hasattr(value, "data"):
            containers.append(value.data)
        if hasattr(value, "_dict"):
            containers.append(value._dict)
        return any(contains_masked_array(item, seen) for item in containers)
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        return any(contains_masked_array(item, seen) for item in value.flat)
    if isinstance(value, Mapping):
        return any(
            contains_masked_array(item, seen) for pair in value.items() for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(contains_masked_array(item, seen) for item in value)
    return False


def _reject_callable_instance_shadows(value: object, name: str) -> None:
    namespace = object.__getattribute__(value, "__dict__")
    shadows = sorted(key for key, item in namespace.items() if callable(item))
    if shadows:
        joined = ", ".join(repr(key) for key in shadows)
        raise TypeError(f"{name} contains callable sparse instance shadow(s): {joined}")


def _storage_error(name: str, detail: str) -> TypeError:
    return TypeError(f"{name} must use trusted internal sparse storage: {detail}")


def _exact_internal_array(
    namespace: dict[str, object],
    field: str,
    name: str,
    *,
    kinds: str,
) -> np.ndarray:
    value = namespace.get(field)
    if type(value) is not np.ndarray:
        raise _storage_error(name, f"{field} must be an exact ndarray")
    if value.dtype.metadata is not None or value.dtype.kind not in kinds:
        raise _storage_error(name, f"{field} has an unsupported dtype")
    return value


def _exact_integer_scalar(value: object) -> bool:
    return type(value) in _EXACT_INTEGER_SCALAR_TYPES


def _exact_numeric_scalar(value: object) -> bool:
    return type(value) in _EXACT_NUMERIC_SCALAR_TYPES


def _validate_trusted_internal_storage(
    value: object,
    name: str,
) -> tuple[str, dict[str, object], tuple[int, int]]:
    """Validate exact non-dispatching storage before traversing sparse content."""

    namespace = object.__getattribute__(value, "__dict__")
    raw_shape = namespace.get("_shape")
    if (
        type(raw_shape) is not tuple
        or len(raw_shape) != 2
        or any(not _exact_integer_scalar(item) for item in raw_shape)
        or any(int(item) < 0 for item in raw_shape)
    ):
        raise _storage_error(name, "_shape must be an exact two-integer tuple")
    shape = (int(raw_shape[0]), int(raw_shape[1]))
    format_name = _SPARSE_FORMAT_BY_TYPE[type(value)]

    if format_name == "coo":
        _exact_internal_array(namespace, "data", name, kinds="biufc")
        coordinates = namespace.get("coords")
        if type(coordinates) is not tuple or len(coordinates) != 2:
            raise _storage_error(name, "coords must be an exact two-array tuple")
        for coordinate in coordinates:
            if (
                type(coordinate) is not np.ndarray
                or coordinate.dtype.metadata is not None
                or coordinate.dtype.kind not in "iu"
                or coordinate.ndim != 1
            ):
                raise _storage_error(
                    name,
                    "COO coordinates must be exact one-dimensional integer ndarrays",
                )
    elif format_name in {"csr", "csc", "bsr"}:
        _exact_internal_array(namespace, "data", name, kinds="biufc")
        _exact_internal_array(namespace, "indices", name, kinds="iu")
        _exact_internal_array(namespace, "indptr", name, kinds="iu")
    elif format_name == "dia":
        _exact_internal_array(namespace, "data", name, kinds="biufc")
        _exact_internal_array(namespace, "offsets", name, kinds="iu")
    elif format_name == "dok":
        storage = namespace.get("_dict")
        if type(storage) is not dict:
            raise _storage_error(name, "DOK _dict must be an exact dict")
        _exact_sparse_dtype(namespace, name)
    elif format_name == "lil":
        rows = _exact_internal_array(namespace, "rows", name, kinds="O")
        data = _exact_internal_array(namespace, "data", name, kinds="O")
        if rows.ndim != 1 or data.ndim != 1 or rows.shape != data.shape:
            raise _storage_error(name, "LIL rows/data must be paired 1D object arrays")
        if rows.shape != (shape[0],):
            raise _storage_error(name, "LIL rows/data must match the row count")
        _exact_sparse_dtype(namespace, name)
    else:  # pragma: no cover - the exact type registry makes this unreachable.
        raise _storage_error(name, "unsupported sparse format")
    return format_name, namespace, shape


def _exact_sparse_dtype(namespace: dict[str, object], name: str) -> np.dtype:
    dtype = namespace.get("dtype")
    if (
        not isinstance(dtype, np.dtype)
        or dtype.metadata is not None
        or dtype.kind not in "biufc"
    ):
        raise _storage_error(name, "dtype must be an exact numeric NumPy dtype")
    return dtype


def _structure_error(name: str, detail: str) -> ValueError:
    return ValueError(f"{name} has invalid sparse structure: {detail}")


def _integer_array_snapshot(
    value: np.ndarray,
    name: str,
    field: str,
    *,
    allow_negative: bool,
) -> np.ndarray:
    if value.ndim != 1:
        raise _structure_error(name, f"{field} must be one-dimensional")
    if value.size:
        minimum = int(np.min(value))
        maximum = int(np.max(value))
        if (not allow_negative and minimum < 0) or maximum > np.iinfo(np.int64).max:
            raise _structure_error(name, f"{field} is outside the supported range")
        if minimum < np.iinfo(np.int64).min:
            raise _structure_error(name, f"{field} is outside the supported range")
    return np.array(value, dtype=np.int64, copy=True, order="C", subok=False)


def _validate_bounds(
    rows: np.ndarray,
    columns: np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> None:
    if np.any((rows < 0) | (rows >= shape[0])) or np.any(
        (columns < 0) | (columns >= shape[1])
    ):
        raise _structure_error(name, "coordinate is out of bounds")


def _validated_indptr(
    value: np.ndarray,
    *,
    major_dimension: int,
    entry_count: int,
    name: str,
) -> np.ndarray:
    pointer = _integer_array_snapshot(
        value,
        name,
        "indptr",
        allow_negative=False,
    )
    if pointer.shape != (major_dimension + 1,):
        raise _structure_error(name, "indptr length does not match the major axis")
    if pointer[0] != 0:
        raise _structure_error(name, "indptr must start at zero")
    if np.any(pointer[1:] < pointer[:-1]):
        raise _structure_error(name, "indptr must be nondecreasing")
    if pointer[-1] != entry_count:
        raise _structure_error(name, "indptr endpoint does not match stored entries")
    return pointer


def _compressed_coordinate_snapshot(
    format_name: str,
    namespace: dict[str, object],
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_source = _exact_internal_array(namespace, "data", name, kinds="biufc")
    indices_source = _exact_internal_array(namespace, "indices", name, kinds="iu")
    indptr_source = _exact_internal_array(namespace, "indptr", name, kinds="iu")
    if data_source.ndim != 1:
        raise _structure_error(name, "compressed data must be one-dimensional")
    indices = _integer_array_snapshot(
        indices_source,
        name,
        "indices",
        allow_negative=False,
    )
    if indices.shape != data_source.shape:
        raise _structure_error(name, "indices and data lengths differ")
    major_dimension = shape[0] if format_name == "csr" else shape[1]
    minor_dimension = shape[1] if format_name == "csr" else shape[0]
    pointer = _validated_indptr(
        indptr_source,
        major_dimension=major_dimension,
        entry_count=data_source.size,
        name=name,
    )
    if np.any(indices >= minor_dimension):
        raise _structure_error(name, "compressed index is out of bounds")
    major = np.repeat(
        np.arange(major_dimension, dtype=np.int64),
        np.diff(pointer),
    )
    rows, columns = (major, indices) if format_name == "csr" else (indices, major)
    data = np.array(data_source, copy=True, order="C", subok=False)
    return data, rows, columns


def _bsr_coordinate_snapshot(
    namespace: dict[str, object],
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_source = _exact_internal_array(namespace, "data", name, kinds="biufc")
    indices_source = _exact_internal_array(namespace, "indices", name, kinds="iu")
    indptr_source = _exact_internal_array(namespace, "indptr", name, kinds="iu")
    if data_source.ndim != 3 or data_source.shape[1] <= 0 or data_source.shape[2] <= 0:
        raise _structure_error(name, "BSR data must contain nonempty 2D blocks")
    block_rows, block_columns = data_source.shape[1:]
    if shape[0] % block_rows or shape[1] % block_columns:
        raise _structure_error(name, "BSR block size must divide the matrix shape")
    indices = _integer_array_snapshot(
        indices_source,
        name,
        "indices",
        allow_negative=False,
    )
    block_count = data_source.shape[0]
    if indices.shape != (block_count,):
        raise _structure_error(name, "BSR indices and block counts differ")
    major_dimension = shape[0] // block_rows
    minor_dimension = shape[1] // block_columns
    pointer = _validated_indptr(
        indptr_source,
        major_dimension=major_dimension,
        entry_count=block_count,
        name=name,
    )
    if np.any(indices >= minor_dimension):
        raise _structure_error(name, "BSR block index is out of bounds")
    stored_block_rows = np.repeat(
        np.arange(major_dimension, dtype=np.int64),
        np.diff(pointer),
    )
    entries_per_block = block_rows * block_columns
    row_offsets = np.repeat(np.arange(block_rows, dtype=np.int64), block_columns)
    column_offsets = np.tile(np.arange(block_columns, dtype=np.int64), block_rows)
    rows = np.repeat(stored_block_rows * block_rows, entries_per_block) + np.tile(
        row_offsets, block_count
    )
    columns = np.repeat(indices * block_columns, entries_per_block) + np.tile(
        column_offsets, block_count
    )
    data = np.array(data_source.reshape(-1), copy=True, order="C", subok=False)
    return data, rows, columns


def _dia_coordinate_snapshot(
    namespace: dict[str, object],
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_source = _exact_internal_array(namespace, "data", name, kinds="biufc")
    offsets_source = _exact_internal_array(namespace, "offsets", name, kinds="iu")
    if data_source.ndim != 2:
        raise _structure_error(name, "DIA data must be two-dimensional")
    offsets = _integer_array_snapshot(
        offsets_source,
        name,
        "offsets",
        allow_negative=True,
    )
    if offsets.shape != (data_source.shape[0],):
        raise _structure_error(name, "DIA offsets and data rows differ")
    if offsets.size:
        if shape[0] == 0 or shape[1] == 0:
            raise _structure_error(name, "empty DIA matrices cannot store offsets")
        if np.any(offsets < -(shape[0] - 1)) or np.any(offsets > shape[1] - 1):
            raise _structure_error(name, "DIA offset is out of bounds")
        if np.unique(offsets).size != offsets.size:
            raise _structure_error(name, "DIA offsets must be unique")
    offset_indices = np.arange(data_source.shape[1], dtype=np.int64)
    row_grid = offset_indices[None, :] - offsets[:, None]
    mask = (
        (row_grid >= 0)
        & (row_grid < shape[0])
        & (offset_indices[None, :] < shape[1])
        & (data_source != 0)
    )
    rows = np.array(row_grid[mask], dtype=np.int64, copy=True)
    column_grid = np.broadcast_to(offset_indices, data_source.shape)
    columns = np.array(column_grid[mask], dtype=np.int64, copy=True)
    data = np.array(data_source[mask], copy=True, order="C", subok=False)
    return data, rows, columns


def _dok_coordinate_snapshot(
    namespace: dict[str, object],
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    storage = namespace["_dict"]
    dtype = _exact_sparse_dtype(namespace, name)
    rows = []
    columns = []
    values = []
    for key, item in dict.items(storage):
        if type(key) is not tuple or len(key) != 2:
            raise _storage_error(name, "DOK entries must use exact scalar keys/values")
        if any(np.ma.isMaskedArray(index) for index in key) or np.ma.isMaskedArray(
            item
        ):
            raise TypeError(f"{name} must not contain masked arrays")
        if any(not _exact_integer_scalar(index) for index in key) or not (
            _exact_numeric_scalar(item)
        ):
            raise _storage_error(name, "DOK entries must use exact scalar keys/values")
        row, column = (int(key[0]), int(key[1]))
        if row < 0 or row >= shape[0] or column < 0 or column >= shape[1]:
            raise _structure_error(name, "DOK coordinate is out of bounds")
        rows.append(row)
        columns.append(column)
        values.append(item)
    return (
        np.asarray(values, dtype=dtype),
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
    )


def _lil_coordinate_snapshot(
    namespace: dict[str, object],
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_storage = namespace["rows"]
    data_storage = namespace["data"]
    dtype = _exact_sparse_dtype(namespace, name)
    rows = []
    columns = []
    values = []
    for row, (row_indices, row_data) in enumerate(
        zip(row_storage, data_storage, strict=True)
    ):
        if (
            type(row_indices) is not list
            or type(row_data) is not list
            or len(row_indices) != len(row_data)
        ):
            raise _storage_error(
                name,
                "LIL rows/data must contain exact lists of scalar entries",
            )
        if any(np.ma.isMaskedArray(index) for index in row_indices) or any(
            np.ma.isMaskedArray(item) for item in row_data
        ):
            raise TypeError(f"{name} must not contain masked arrays")
        if any(not _exact_integer_scalar(index) for index in row_indices) or any(
            not _exact_numeric_scalar(item) for item in row_data
        ):
            raise _storage_error(
                name,
                "LIL rows/data must contain exact lists of scalar entries",
            )
        integer_indices = [int(index) for index in row_indices]
        if any(index < 0 or index >= shape[1] for index in integer_indices):
            raise _structure_error(name, "LIL column index is out of bounds")
        if any(
            right <= left for left, right in zip(integer_indices, integer_indices[1:])
        ):
            raise _structure_error(name, "LIL row indices must be strictly increasing")
        rows.extend([row] * len(integer_indices))
        columns.extend(integer_indices)
        values.extend(row_data)
    return (
        np.asarray(values, dtype=dtype),
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
    )


def _native_coordinate_snapshot(
    format_name: str,
    namespace: dict[str, object],
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if format_name == "coo":
        data_source = namespace["data"]
        row_source, column_source = namespace["coords"]
        if data_source.ndim != 1:
            raise _structure_error(name, "COO data must be one-dimensional")
        rows = _integer_array_snapshot(
            row_source,
            name,
            "row coordinates",
            allow_negative=False,
        )
        columns = _integer_array_snapshot(
            column_source,
            name,
            "column coordinates",
            allow_negative=False,
        )
        if rows.shape != columns.shape or rows.shape != data_source.shape:
            raise _structure_error(name, "COO coordinate and data lengths differ")
        data = np.array(data_source, copy=True, order="C", subok=False)
    elif format_name in {"csr", "csc"}:
        data, rows, columns = _compressed_coordinate_snapshot(
            format_name,
            namespace,
            shape,
            name,
        )
    elif format_name == "bsr":
        data, rows, columns = _bsr_coordinate_snapshot(namespace, shape, name)
    elif format_name == "dia":
        data, rows, columns = _dia_coordinate_snapshot(namespace, shape, name)
    elif format_name == "dok":
        data, rows, columns = _dok_coordinate_snapshot(namespace, shape, name)
    elif format_name == "lil":
        data, rows, columns = _lil_coordinate_snapshot(namespace, shape, name)
    else:  # pragma: no cover - exact type validation makes this unreachable.
        raise _structure_error(name, "unsupported sparse format")
    _validate_bounds(rows, columns, shape, name)
    if data.ndim != 1 or rows.shape != columns.shape or rows.shape != data.shape:
        raise _structure_error(name, "coordinate and data arrays are inconsistent")
    if data.size >= 2:
        order = np.lexsort((columns, rows))
        ordered_rows = rows[order]
        ordered_columns = columns[order]
        if np.any(
            (ordered_rows[1:] == ordered_rows[:-1])
            & (ordered_columns[1:] == ordered_columns[:-1])
        ):
            raise _structure_error(name, "duplicate coordinates are not supported")
    return data, rows, columns


def sparse_coordinate_snapshot(
    value: object,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Return validated, independent COO data/row/column snapshots.

    Exact native storage and structural invariants are validated before any
    traversal. Coordinates are derived directly from independent native-array
    snapshots; no SciPy conversion method or C conversion helper is invoked.
    """

    if not sparse.issparse(value) or type(value) not in SUPPORTED_SPARSE_TYPES:
        raise TypeError(f"{name} must use an exact supported SciPy sparse type")
    _reject_callable_instance_shadows(value, name)
    format_name, namespace, shape = _validate_trusted_internal_storage(value, name)
    data, rows, columns = _native_coordinate_snapshot(
        format_name,
        namespace,
        shape,
        name,
    )
    return data, rows, columns, shape


__all__ = [
    "SUPPORTED_SPARSE_TYPES",
    "contains_masked_array",
    "sparse_coordinate_snapshot",
]
