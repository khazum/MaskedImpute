"""Leakage-resistant immutable contracts for benchmark method execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

from maskimpute.sparse_input import contains_masked_array, sparse_coordinate_snapshot

if TYPE_CHECKING:
    import anndata as ad


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_ALLOWED_UNS = frozenset(
    {"source_dataset_sha256", "normalization", "allowed_covariates"}
)
_TRUTH_NAME = re.compile(
    r"(?:^|_)(?:truth|label|cluster|cell_?type|condition|outcome|phenotype|"
    r"pseudotime|trajectory|lineage|marker)(?:_|$)",
    flags=re.IGNORECASE,
)


class MethodContractError(ValueError):
    """Raised when a method registry, input, output, or run record is invalid."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Immutable upstream source or in-tree freeze binding."""

    kind: str
    url: str | None
    revision: str | None
    tree: str | None
    cache_path: str | None
    freeze_binding: str | None


@dataclass(frozen=True, slots=True)
class LicenseSpec:
    """Repository-license status without inferred permissions."""

    status: str
    spdx: str | None
    notice: str | None


@dataclass(frozen=True, slots=True)
class CitationSpec:
    """Verified DOI or an explicit pending citation record."""

    status: str
    doi: str | None
    url: str | None


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Execution environment identity and non-fabricated lock status."""

    id: str
    status: str
    lock_sha256: str | None


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Prespecified per-run resource ceiling."""

    timeout_seconds: int
    cpu_cores: int
    gpu_required: bool
    max_rss_gib: int | float
    max_gpu_gib: int | float

    @property
    def gpu_mode(self) -> str:
        """Return the closed GPU scheduling mode used by the runner budget."""

        if self.gpu_required:
            return "required"
        return "forbidden"


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One fully declared method in the comparison denominator."""

    id: str
    display_name: str
    role: str
    track: str
    execution_scope: str
    applicability_reason: str | None
    input_scale: str
    output_scale: str
    stochastic: bool
    seed_policy: str
    source: SourceSpec
    license: LicenseSpec
    citation: CitationSpec
    environment: EnvironmentSpec
    resources: ResourceSpec
    preserves_observed_positives: bool
    source_policy: str
    integration_status: str
    integration_reason: str | None

    @property
    def executable(self) -> bool:
        """Whether a planner may schedule this method in its declared scope."""

        return self.execution_scope in {
            "same_input_required",
            "external_reference_only",
        }


@dataclass(frozen=True, slots=True)
class CovariateColumn:
    """Immutable values and dtype semantics for one allowed covariate."""

    name: str
    kind: str
    dtype: str
    values: tuple[str | bool | int | float, ...]
    categories: tuple[str | bool | int | float, ...] = ()
    ordered: bool = False
    codes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class MethodInput:
    """Truth-free count snapshot passed across a method adapter boundary."""

    source_dataset_sha256: str
    obs_ids: tuple[str, ...]
    var_ids: tuple[str, ...]
    shape: tuple[int, int]
    obs_covariates: tuple[CovariateColumn, ...]
    var_covariates: tuple[CovariateColumn, ...]
    _count_bytes: bytes = field(repr=False)
    _normalization_bytes: bytes = field(repr=False)

    @property
    def counts(self) -> np.ndarray:
        """Return a read-only float64 view backed by immutable bytes."""

        return np.frombuffer(self._count_bytes, dtype="<f8").reshape(self.shape)

    @property
    def normalization(self) -> object:
        """Return a fresh JSON value describing the declared input normalization."""

        return json.loads(self._normalization_bytes.decode("utf-8"))

    def covariate_frame(self, axis: str) -> pd.DataFrame:
        """Reconstruct a defensive DataFrame with categorical semantics intact."""

        if axis == "obs":
            identifiers = self.obs_ids
            columns = self.obs_covariates
        elif axis == "var":
            identifiers = self.var_ids
            columns = self.var_covariates
        else:
            raise ValueError("axis must be obs or var")
        frame = pd.DataFrame(index=pd.Index(identifiers, dtype=object))
        for column in columns:
            if column.kind == "categorical":
                values = pd.Categorical.from_codes(
                    column.codes,
                    categories=column.categories,
                    ordered=column.ordered,
                )
                frame[column.name] = pd.Series(values, index=frame.index)
            else:
                frame[column.name] = pd.Series(
                    column.values,
                    index=frame.index,
                    dtype=column.dtype,
                )
        return frame


@dataclass(frozen=True, slots=True)
class MethodOutputSnapshot:
    """Immutable aligned output bound to its method and source dataset."""

    method_id: str
    source_dataset_sha256: str
    output_scale: str
    obs_ids: tuple[str, ...]
    var_ids: tuple[str, ...]
    shape: tuple[int, int]
    matrix_sha256: str
    _matrix_bytes: bytes = field(repr=False)

    @property
    def matrix(self) -> np.ndarray:
        """Return a read-only float64 view backed by immutable bytes."""

        return np.frombuffer(self._matrix_bytes, dtype="<f8").reshape(self.shape)


@dataclass(frozen=True, slots=True)
class MethodRunRecord:
    """Closed, measured record for one completed or failed method attempt."""

    schema_version: int
    run_id: str
    method_id: str
    source_dataset_sha256: str
    status: str
    seed: int | None
    runtime_seconds: int | float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    output_sha256: str | None
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the exact canonical serialization payload."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "method_id": self.method_id,
            "source_dataset_sha256": self.source_dataset_sha256,
            "status": self.status,
            "seed": self.seed,
            "runtime_seconds": self.runtime_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "output_sha256": self.output_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MethodStatusRow:
    """Denominator-preserving summary row, including failures and pending methods."""

    method_id: str
    status: str
    reason: str | None
    run_count: int


def _reject_inexact_large_integers(values: np.ndarray, name: str) -> None:
    if values.dtype.kind not in {"i", "u"}:
        return
    flattened = values.reshape(-1)
    for value in flattened:
        integer = int(value)
        if int(float(integer)) != integer:
            raise MethodContractError(
                f"{name} integer values must be exactly representable as float64"
            )


def _matrix_values(matrix: object, name: str) -> np.ndarray:
    if sparse.issparse(matrix):
        try:
            data, rows, columns, shape = sparse_coordinate_snapshot(matrix, name)
        except (TypeError, ValueError) as error:
            raise MethodContractError(str(error)) from error
        source_values = data
        if source_values.dtype.metadata is not None:
            raise MethodContractError(f"{name} must not use dtype metadata")
        if source_values.dtype.kind == "b":
            raise MethodContractError(f"{name} must not contain boolean values")
        if (
            source_values.dtype.kind not in {"i", "u", "f"}
            or source_values.dtype.itemsize > 8
        ):
            raise MethodContractError(
                f"{name} must use native numeric values up to float64"
            )
        _reject_inexact_large_integers(source_values, name)
        try:
            array = np.zeros(shape, dtype=np.float64)
            if source_values.size:
                np.add.at(
                    array,
                    (rows, columns),
                    np.asarray(source_values, dtype=np.float64),
                )
        except (TypeError, ValueError, OverflowError) as error:
            raise MethodContractError(
                f"{name} must fit finite float64 values"
            ) from error
        return array
    if np.ma.isMaskedArray(matrix):
        raise MethodContractError(f"{name} must not contain masked arrays")
    if type(matrix) is not np.ndarray:
        raise MethodContractError(
            f"{name} must use an exact ndarray or exact supported SciPy sparse type"
        )
    if contains_masked_array(matrix):
        raise MethodContractError(f"{name} must not contain masked arrays")
    if matrix.ndim != 2:
        raise MethodContractError(f"{name} must be a numeric two-dimensional matrix")
    if matrix.dtype.metadata is not None:
        raise MethodContractError(f"{name} must not use dtype metadata")
    if matrix.dtype.kind == "b":
        raise MethodContractError(f"{name} must not contain boolean values")
    if matrix.dtype.kind not in {"i", "u", "f"} or matrix.dtype.itemsize > 8:
        raise MethodContractError(
            f"{name} must use native numeric values up to float64"
        )
    _reject_inexact_large_integers(matrix, name)
    try:
        return np.array(matrix, dtype=np.float64, copy=True, order="C", subok=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise MethodContractError(f"{name} must fit finite float64 values") from error


def _validate_ids(values: Sequence[object], axis: str) -> tuple[str, ...]:
    identifiers = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise MethodContractError(f"{axis} IDs must be nonempty strings")
    if len(identifiers) != len(set(identifiers)):
        raise MethodContractError(f"{axis} IDs must be unique")
    return identifiers


def _canonical_json_value(
    value: object,
    name: str,
    active: set[int] | None = None,
) -> object:
    if np.ma.isMaskedArray(value):
        raise MethodContractError(f"{name} must not contain masked arrays")
    if isinstance(value, np.generic):
        return _canonical_json_value(value.item(), name, active)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MethodContractError(f"{name} must contain finite JSON numbers")
        return value
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise MethodContractError(f"{name} must not contain recursive JSON values")
    if isinstance(value, np.ndarray):
        if value.dtype.metadata is not None:
            raise MethodContractError(f"{name} must not use dtype metadata")
        active.add(identity)
        try:
            return _canonical_json_value(value.tolist(), name, active)
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise MethodContractError(f"{name} JSON object keys must be strings")
        active.add(identity)
        try:
            return {
                key: _canonical_json_value(nested, name, active)
                for key, nested in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        active.add(identity)
        try:
            return [_canonical_json_value(nested, name, active) for nested in value]
        finally:
            active.remove(identity)
    raise MethodContractError(f"{name} must be canonical JSON")


def _covariate_names(declaration: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    declaration = _canonical_json_value(declaration, "allowed_covariates")
    if isinstance(declaration, list):
        obs_value: object = declaration
        var_value: object = []
    elif isinstance(declaration, Mapping):
        if set(declaration) - {"obs", "var"}:
            raise MethodContractError("allowed_covariates has unknown keys")
        obs_value = declaration.get("obs", [])
        var_value = declaration.get("var", [])
    else:
        raise MethodContractError("allowed_covariates must declare obs and var columns")

    result: list[tuple[str, ...]] = []
    for axis, value in (("obs", obs_value), ("var", var_value)):
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or any(not isinstance(name, str) or not name for name in value)
        ):
            raise MethodContractError(
                f"allowed_covariates {axis} must be a list of column names"
            )
        names = tuple(value)
        if len(names) != len(set(names)):
            raise MethodContractError(f"allowed_covariates {axis} contains duplicates")
        if any(_TRUTH_NAME.search(name) for name in names):
            raise MethodContractError(
                f"allowed_covariates {axis} contains evaluator metadata"
            )
        result.append(names)
    return result[0], result[1]


def _covariate_scalar(value: object, label: str) -> str | bool | int | float:
    normalized = _canonical_json_value(value, label)
    if not isinstance(normalized, (str, bool, int, float)):
        raise MethodContractError(f"{label} must contain immutable scalar values")
    return normalized


def _snapshot_covariates(
    frame: pd.DataFrame,
    names: tuple[str, ...],
    axis: str,
) -> tuple[CovariateColumn, ...]:
    snapshots = []
    for name in names:
        series = frame[name]
        label = f"allowed_covariates {axis}.{name}"
        if isinstance(series.dtype, pd.CategoricalDtype):
            codes = tuple(int(value) for value in series.cat.codes.tolist())
            if any(code < 0 for code in codes):
                raise MethodContractError(f"{label} must not contain missing values")
            categories = tuple(
                _covariate_scalar(value, label)
                for value in series.cat.categories.tolist()
            )
            values = tuple(categories[code] for code in codes)
            snapshots.append(
                CovariateColumn(
                    name=name,
                    kind="categorical",
                    dtype=str(series.dtype),
                    values=values,
                    categories=categories,
                    ordered=bool(series.cat.ordered),
                    codes=codes,
                )
            )
            continue
        if bool(series.isna().any()):
            raise MethodContractError(f"{label} must not contain missing values")
        if pd.api.types.is_object_dtype(series.dtype) or isinstance(
            series.dtype, pd.StringDtype
        ):
            raw_values = series.tolist()
            if any(not isinstance(value, str) for value in raw_values):
                raise MethodContractError(f"{label} must contain scalar strings")
            values = tuple(raw_values)
        elif pd.api.types.is_bool_dtype(series.dtype):
            values = tuple(bool(value) for value in series.tolist())
        elif pd.api.types.is_integer_dtype(series.dtype):
            values = tuple(int(value) for value in series.tolist())
        elif pd.api.types.is_float_dtype(series.dtype):
            values = tuple(float(value) for value in series.tolist())
            if any(not math.isfinite(value) for value in values):
                raise MethodContractError(f"{label} must contain finite values")
        else:
            raise MethodContractError(f"{label} has an unsupported covariate dtype")
        snapshots.append(
            CovariateColumn(
                name=name,
                kind="plain",
                dtype=str(series.dtype),
                values=values,
            )
        )
    return tuple(snapshots)


def _validate_closed_inference_slots(adata: ad.AnnData) -> None:
    occupied = []
    for name in ("layers", "obsm", "varm", "obsp", "varp"):
        if len(getattr(adata, name)):
            occupied.append(name)
    if adata.raw is not None:
        occupied.append("raw")
    unknown_uns = set(adata.uns) - _ALLOWED_UNS
    if occupied or unknown_uns:
        details = sorted([*occupied, *(f"uns.{key}" for key in unknown_uns)])
        raise MethodContractError(
            "method input must be a truth-free closed slot view; found "
            + ", ".join(details)
        )


def prepare_method_input(adata: ad.AnnData) -> MethodInput:
    """Validate and snapshot an evaluator-created truth-free inference AnnData."""

    import anndata as ad_module

    if not isinstance(adata, ad_module.AnnData):
        raise TypeError("method input must be an AnnData object")
    _validate_closed_inference_slots(adata)
    source_sha256 = adata.uns.get("source_dataset_sha256")
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise MethodContractError(
            "method input source_dataset_sha256 must be lowercase SHA-256"
        )
    obs_ids = _validate_ids(adata.obs_names.tolist(), "obs")
    var_ids = _validate_ids(adata.var_names.tolist(), "var")
    declaration = adata.uns.get("allowed_covariates", {"obs": [], "var": []})
    obs_covariates, var_covariates = _covariate_names(declaration)
    if not adata.obs.columns.is_unique or not adata.var.columns.is_unique:
        raise MethodContractError("method input covariate column names must be unique")
    observed_obs = tuple(adata.obs.columns.tolist())
    observed_var = tuple(adata.var.columns.tolist())
    if observed_obs != obs_covariates:
        raise MethodContractError("method input contains undeclared obs columns")
    if observed_var != var_covariates:
        raise MethodContractError("method input contains undeclared var columns")
    normalization = _canonical_json_value(
        adata.uns.get("normalization"), "normalization"
    )
    normalization_bytes = json.dumps(
        normalization,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    obs_snapshots = _snapshot_covariates(adata.obs, obs_covariates, "obs")
    var_snapshots = _snapshot_covariates(adata.var, var_covariates, "var")

    counts = _matrix_values(adata.X, "method input counts")
    if counts.shape != adata.shape:
        raise MethodContractError("method input count shape is inconsistent")
    if not np.isfinite(counts).all():
        raise MethodContractError("method input counts must be finite")
    if bool((counts < 0).any()):
        raise MethodContractError("method input counts must be nonnegative")
    if not bool((counts == np.floor(counts)).all()):
        raise MethodContractError("method input counts must contain integers")
    return MethodInput(
        source_dataset_sha256=source_sha256,
        obs_ids=obs_ids,
        var_ids=var_ids,
        shape=tuple(adata.shape),
        obs_covariates=obs_snapshots,
        var_covariates=var_snapshots,
        _count_bytes=np.asarray(counts, dtype="<f8", order="C").tobytes(order="C"),
        _normalization_bytes=normalization_bytes,
    )


def _output_digest(
    *,
    method_id: str,
    source_dataset_sha256: str,
    output_scale: str,
    obs_ids: tuple[str, ...],
    var_ids: tuple[str, ...],
    shape: tuple[int, int],
    matrix_bytes: bytes,
) -> str:
    digest = hashlib.sha256()
    binding = json.dumps(
        {
            "method_id": method_id,
            "source_dataset_sha256": source_dataset_sha256,
            "output_scale": output_scale,
            "obs_ids": obs_ids,
            "var_ids": var_ids,
            "shape": shape,
            "dtype": "<f8",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(binding)
    digest.update(matrix_bytes)
    return digest.hexdigest()


def snapshot_method_output(
    spec: MethodSpec,
    method_input: MethodInput,
    matrix: object,
    *,
    source_dataset_sha256: str,
    output_scale: str,
    obs_ids: Sequence[str],
    var_ids: Sequence[str],
) -> MethodOutputSnapshot:
    """Validate and freeze one aligned method output matrix."""

    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if source_dataset_sha256 != method_input.source_dataset_sha256:
        raise MethodContractError("output source dataset binding does not match input")
    if output_scale != spec.output_scale:
        raise MethodContractError("output scale does not match method specification")
    output_obs_ids = tuple(obs_ids)
    output_var_ids = tuple(var_ids)
    if output_obs_ids != method_input.obs_ids:
        raise MethodContractError("output obs IDs do not match method input")
    if output_var_ids != method_input.var_ids:
        raise MethodContractError("output var IDs do not match method input")

    output = _matrix_values(matrix, "method output")
    if output.shape != method_input.shape:
        raise MethodContractError(
            f"method output shape {output.shape} does not match {method_input.shape}"
        )
    if not np.isfinite(output).all():
        raise MethodContractError("method output must contain finite values")
    if bool((output < 0).any()):
        raise MethodContractError("method output must be nonnegative")
    if spec.preserves_observed_positives:
        counts = method_input.counts
        positive = counts > 0
        if not np.array_equal(output[positive], counts[positive]):
            raise MethodContractError("method output must preserve observed positives")

    matrix_bytes = np.asarray(output, dtype="<f8", order="C").tobytes(order="C")
    digest = _output_digest(
        method_id=spec.id,
        source_dataset_sha256=source_dataset_sha256,
        output_scale=output_scale,
        obs_ids=output_obs_ids,
        var_ids=output_var_ids,
        shape=method_input.shape,
        matrix_bytes=matrix_bytes,
    )
    return MethodOutputSnapshot(
        method_id=spec.id,
        source_dataset_sha256=source_dataset_sha256,
        output_scale=output_scale,
        obs_ids=output_obs_ids,
        var_ids=output_var_ids,
        shape=method_input.shape,
        matrix_sha256=digest,
        _matrix_bytes=matrix_bytes,
    )


_RUN_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "method_id",
        "source_dataset_sha256",
        "status",
        "seed",
        "runtime_seconds",
        "peak_rss_bytes",
        "peak_gpu_bytes",
        "stdout_sha256",
        "stderr_sha256",
        "output_sha256",
        "reason",
    }
)


def _sha256(value: object, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise MethodContractError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_number(value: object, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise MethodContractError(f"{name} must be finite and nonnegative")
    return value


def validate_run_record(
    registry: object, payload: Mapping[str, object]
) -> MethodRunRecord:
    """Validate one status-complete run payload against its method seed policy."""

    from .registry import MethodRegistry

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(payload, Mapping):
        raise TypeError("run record must be a mapping")
    if set(payload) != _RUN_KEYS:
        missing = sorted(_RUN_KEYS - set(payload))
        extra = sorted(set(payload) - _RUN_KEYS)
        raise MethodContractError(
            f"run record fields are not closed; missing={missing}, extra={extra}"
        )
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise MethodContractError("run record schema_version must be 1")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
        raise MethodContractError("run_id must be a safe lowercase identifier")
    method_id = payload["method_id"]
    if not isinstance(method_id, str):
        raise MethodContractError("method_id must be a string")
    try:
        spec = registry.by_id(method_id)
    except KeyError as error:
        raise MethodContractError(f"unknown method_id: {method_id}") from error
    source_sha256 = _sha256(payload["source_dataset_sha256"], "source_dataset_sha256")
    assert source_sha256 is not None
    status = payload["status"]
    if status not in {"completed", "failed"}:
        raise MethodContractError("run status must be completed or failed")
    seed = payload["seed"]
    if spec.seed_policy == "required":
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise MethodContractError(
                "stochastic method seed must be an integer in [0, 2^63)"
            )
    elif seed is not None:
        raise MethodContractError("deterministic method seed must be null")
    runtime = _nonnegative_number(payload["runtime_seconds"], "runtime_seconds")
    for name in ("peak_rss_bytes", "peak_gpu_bytes"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MethodContractError(f"{name} must be a nonnegative integer")
    stdout_sha256 = _sha256(payload["stdout_sha256"], "stdout_sha256")
    stderr_sha256 = _sha256(payload["stderr_sha256"], "stderr_sha256")
    assert stdout_sha256 is not None and stderr_sha256 is not None
    reason = payload["reason"]
    if status == "completed":
        output_sha256 = _sha256(payload["output_sha256"], "completed output_sha256")
        if reason is not None:
            raise MethodContractError("completed run reason must be null")
    else:
        if payload["output_sha256"] is not None:
            raise MethodContractError("failed run output_sha256 must be null")
        output_sha256 = None
        if not isinstance(reason, str) or not reason.strip():
            raise MethodContractError("failed run reason must be a nonempty string")

    return MethodRunRecord(
        schema_version=1,
        run_id=run_id,
        method_id=method_id,
        source_dataset_sha256=source_sha256,
        status=status,
        seed=seed,
        runtime_seconds=runtime,
        peak_rss_bytes=payload["peak_rss_bytes"],
        peak_gpu_bytes=payload["peak_gpu_bytes"],
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
        output_sha256=output_sha256,
        reason=reason,
    )


def canonical_run_record_bytes(record: MethodRunRecord) -> bytes:
    """Serialize a validated run record deterministically."""

    if not isinstance(record, MethodRunRecord):
        raise TypeError("record must be a MethodRunRecord")
    return (
        json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def build_method_status_table(
    registry: object,
    records: Sequence[MethodRunRecord],
) -> tuple[MethodStatusRow, ...]:
    """Keep every registered method in status output, including failed/pending rows."""

    from .registry import MethodRegistry

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    grouped: dict[str, list[MethodRunRecord]] = {
        method_id: [] for method_id in registry.ids
    }
    seen_runs: set[str] = set()
    for record in records:
        if not isinstance(record, MethodRunRecord):
            raise TypeError("records must contain MethodRunRecord values")
        if record.method_id not in grouped:
            raise MethodContractError(
                f"run references unknown method: {record.method_id}"
            )
        if record.run_id in seen_runs:
            raise MethodContractError(f"duplicate run_id: {record.run_id}")
        seen_runs.add(record.run_id)
        grouped[record.method_id].append(record)

    rows = []
    for spec in registry.methods:
        method_records = grouped[spec.id]
        failures = [record for record in method_records if record.status == "failed"]
        if failures:
            status = "failed"
            reason = ";".join(
                sorted({record.reason or "failed" for record in failures})
            )
        elif method_records:
            status = "completed"
            reason = None
        elif not spec.executable:
            status = spec.execution_scope
            reason = spec.applicability_reason
        else:
            status = spec.integration_status
            reason = spec.integration_reason
        rows.append(
            MethodStatusRow(
                method_id=spec.id,
                status=status,
                reason=reason,
                run_count=len(method_records),
            )
        )
    return tuple(rows)


def immutable_status_mapping(rows: Sequence[MethodStatusRow]) -> Mapping[str, str]:
    """Return a read-only status lookup for downstream table code."""

    return MappingProxyType({row.method_id: row.status for row in rows})


__all__ = [
    "CitationSpec",
    "CovariateColumn",
    "EnvironmentSpec",
    "LicenseSpec",
    "MethodContractError",
    "MethodInput",
    "MethodOutputSnapshot",
    "MethodRunRecord",
    "MethodSpec",
    "MethodStatusRow",
    "ResourceSpec",
    "SourceSpec",
    "build_method_status_table",
    "canonical_run_record_bytes",
    "prepare_method_input",
    "snapshot_method_output",
    "validate_run_record",
]
