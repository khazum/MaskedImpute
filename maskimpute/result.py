"""Result containers for MaskImpute inference."""

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse


def _freeze_array(value: Any) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _freeze_matrix(value: Any) -> Any:
    if sparse.issparse(value):
        result = value.tocsr(copy=True)
        result.sum_duplicates()
        result.sort_indices()
        result.data.flags.writeable = False
        result.indices.flags.writeable = False
        result.indptr.flags.writeable = False
        return result
    return _freeze_array(value)


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


def _validate_diagnostic(value: Any, path: str = "diagnostics") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_diagnostic(item, f"{path}.{key}")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError(f"{path} cannot contain object arrays")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{path} must contain only finite values")
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _validate_diagnostic(item, f"{path}[{index}]")
        return
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be finite")
        return
    if value is None or isinstance(value, (str, bytes, bool)):
        return
    raise TypeError(f"{path} contains an unsupported value")


def _freeze_diagnostic(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_diagnostic(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_diagnostic(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_diagnostic(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class ImputationResult:
    """Outputs from a MaskImpute inference run."""

    selective_counts: Any
    denoised_counts: Any
    p_pre_zero: Any
    latent: Any
    diagnostics: Any

    def __post_init__(self) -> None:
        for name in ("selective_counts", "denoised_counts"):
            object.__setattr__(self, name, _freeze_matrix(getattr(self, name)))
        for name in ("p_pre_zero", "latent"):
            if sparse.issparse(getattr(self, name)):
                raise TypeError(f"{name} must be a dense matrix")
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))

        _validate_real_array(self.selective_counts, "selective_counts", nonnegative=True)
        _validate_real_array(self.denoised_counts, "denoised_counts", nonnegative=True)
        _validate_real_array(self.p_pre_zero, "p_pre_zero", nonnegative=True)
        _validate_real_array(self.latent, "latent", nonnegative=False)
        if self.selective_counts.shape != self.denoised_counts.shape:
            raise ValueError("selective_counts and denoised_counts shapes must match")
        if self.p_pre_zero.shape != self.selective_counts.shape:
            raise ValueError("p_pre_zero shape must match count matrices")
        if self.latent.shape[0] != self.selective_counts.shape[0]:
            raise ValueError("latent rows must match count-matrix rows")
        if np.any(self.p_pre_zero > 1):
            raise ValueError("p_pre_zero probabilities must lie in [0, 1]")

        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        _validate_diagnostic(self.diagnostics)
        object.__setattr__(self, "diagnostics", _freeze_diagnostic(self.diagnostics))
