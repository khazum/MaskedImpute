"""Strict method registry and leakage-safe benchmark execution contracts."""

from types import MappingProxyType

import numpy as np

from .base import (
    _output_digest,
    CitationSpec,
    CovariateColumn,
    EnvironmentSpec,
    LicenseSpec,
    MethodContractError,
    MethodInput,
    MethodOutputSnapshot,
    MethodRunRecord,
    MethodSpec,
    MethodStatusRow,
    ResourceSpec,
    SourceSpec,
    build_method_status_table,
    canonical_run_record_bytes,
    prepare_method_input,
    snapshot_method_output,
    validate_run_record,
)
from .alra import (
    ALRAConfig,
    alra_to_evaluator_counts,
    finalize_alra_output,
    run_alra,
)
from .dca import DCAConfig, dca_to_evaluator_counts, finalize_dca_output, run_dca
from .magic import (
    MAGICConfig,
    finalize_magic_output,
    magic_to_evaluator_counts,
    run_magic,
)
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    SourceReceipt,
    count_equivalent_to_log2_cp10k,
    log1p_cp10k,
    observed_to_evaluator_counts,
    run_observed,
    verify_pinned_source,
)
from .registry import (
    MethodRegistry,
    load_method_registry,
    verify_cached_method_sources,
)
from .saver import (
    SAVERConfig,
    finalize_saver_output,
    run_saver,
    saver_to_evaluator_counts,
)
from .scvi import (
    SCVIConfig,
    finalize_scvi_output,
    frequencies_to_observed_library_counts,
    run_scvi,
    scvi_to_evaluator_counts,
)


CORE_EVALUATOR_COUNT_CONVERTERS = MappingProxyType(
    {
        "observed": observed_to_evaluator_counts,
        "alra": alra_to_evaluator_counts,
        "magic": magic_to_evaluator_counts,
        "dca": dca_to_evaluator_counts,
        "scvi": scvi_to_evaluator_counts,
        "saver": saver_to_evaluator_counts,
    }
)
CORE_EVALUATOR_NATIVE_SCALES = MappingProxyType(
    {
        "observed": "raw_counts",
        "alra": "log1p_cp10k",
        "magic": "log1p_cp10k",
        "dca": "raw_counts",
        "scvi": "raw_counts",
        "saver": "method_native_normalized",
    }
)


def core_output_to_evaluator_counts(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Validate a bound native snapshot and apply its declared count conversion."""

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if not isinstance(snapshot, MethodOutputSnapshot):
        raise TypeError("snapshot must be a MethodOutputSnapshot")
    try:
        converter = CORE_EVALUATOR_COUNT_CONVERTERS[snapshot.method_id]
        expected_scale = CORE_EVALUATOR_NATIVE_SCALES[snapshot.method_id]
    except KeyError as error:
        raise ValueError(
            f"no evaluator count converter is declared for method {snapshot.method_id}"
        ) from error
    if snapshot.source_dataset_sha256 != method_input.source_dataset_sha256:
        raise ValueError("snapshot source dataset does not match the method input")
    if snapshot.shape != method_input.shape:
        raise ValueError("snapshot shape does not match the method input")
    if snapshot.obs_ids != method_input.obs_ids:
        raise ValueError("snapshot cell IDs do not match the method input")
    if snapshot.var_ids != method_input.var_ids:
        raise ValueError("snapshot gene IDs do not match the method input")
    if snapshot.output_scale != expected_scale:
        raise ValueError(
            f"snapshot native output scale must be {expected_scale} for {snapshot.method_id}"
        )
    try:
        native_output = snapshot.matrix
    except ValueError as error:
        raise ValueError("snapshot matrix bytes do not match its shape") from error
    matrix_bytes = np.asarray(native_output, dtype="<f8", order="C").tobytes(order="C")
    expected_hash = _output_digest(
        method_id=snapshot.method_id,
        source_dataset_sha256=snapshot.source_dataset_sha256,
        output_scale=snapshot.output_scale,
        obs_ids=snapshot.obs_ids,
        var_ids=snapshot.var_ids,
        shape=snapshot.shape,
        matrix_bytes=matrix_bytes,
    )
    if snapshot.matrix_sha256 != expected_hash:
        raise ValueError("snapshot matrix hash does not match its bound content")
    return converter(method_input, native_output)


def core_output_to_evaluator_log2_cp10k(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Convert one declared core output to the common log2(CP10k+1) scale."""

    counts = core_output_to_evaluator_counts(method_input, snapshot)
    return count_equivalent_to_log2_cp10k(counts)


__all__ = [
    "ALRAConfig",
    "AdapterExecution",
    "AdapterUnavailableError",
    "CitationSpec",
    "CompatibilityEvent",
    "CORE_EVALUATOR_COUNT_CONVERTERS",
    "CORE_EVALUATOR_NATIVE_SCALES",
    "CovariateColumn",
    "DCAConfig",
    "EnvironmentSpec",
    "LicenseSpec",
    "MethodContractError",
    "MethodInput",
    "MethodOutputSnapshot",
    "MethodRegistry",
    "MethodRunRecord",
    "MethodSpec",
    "MethodStatusRow",
    "MAGICConfig",
    "ResourceSpec",
    "SAVERConfig",
    "SCVIConfig",
    "SourceSpec",
    "SourceReceipt",
    "build_method_status_table",
    "canonical_run_record_bytes",
    "core_output_to_evaluator_counts",
    "core_output_to_evaluator_log2_cp10k",
    "count_equivalent_to_log2_cp10k",
    "dca_to_evaluator_counts",
    "alra_to_evaluator_counts",
    "finalize_alra_output",
    "finalize_dca_output",
    "finalize_magic_output",
    "finalize_saver_output",
    "finalize_scvi_output",
    "frequencies_to_observed_library_counts",
    "load_method_registry",
    "log1p_cp10k",
    "magic_to_evaluator_counts",
    "observed_to_evaluator_counts",
    "prepare_method_input",
    "snapshot_method_output",
    "run_alra",
    "run_dca",
    "run_magic",
    "run_observed",
    "run_saver",
    "run_scvi",
    "saver_to_evaluator_counts",
    "scvi_to_evaluator_counts",
    "validate_run_record",
    "verify_pinned_source",
    "verify_cached_method_sources",
]
