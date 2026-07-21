"""Strict method registry and leakage-safe benchmark execution contracts."""

from collections.abc import Callable, Mapping
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
    finalize_alra_direct_output,
    finalize_alra_output,
    run_alra,
    run_alra_direct,
)
from .afmf import (
    AFMFConfig,
    afmf_to_evaluator_counts,
    finalize_afmf_direct_output,
    finalize_afmf_output,
    run_afmf,
    run_afmf_direct,
)
from .biaeimpute import (
    BiAEImputeConfig,
    biaeimpute_to_evaluator_counts,
    finalize_biaeimpute_direct_output,
    finalize_biaeimpute_output,
    run_biaeimpute,
    run_biaeimpute_direct,
)
from .dca import (
    DCAConfig,
    dca_to_evaluator_counts,
    finalize_dca_direct_output,
    finalize_dca_output,
    run_dca,
    run_dca_direct,
)
from .direct import (
    DirectAdapterExecution,
    DirectMethodOutput,
    finalize_direct_method_output,
)
from .d3impute import (
    D3ImputeConfig,
    MatchedBulkReference,
    d3impute_to_evaluator_counts,
    finalize_d3impute_output,
    prepare_matched_bulk_reference,
    run_d3impute,
    validate_matched_bulk_reference,
)
from .magic import (
    MAGICConfig,
    finalize_magic_direct_output,
    finalize_magic_output,
    magic_to_evaluator_counts,
    run_magic,
    run_magic_direct,
)
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    SourceReceipt,
    count_equivalent_to_log2_cp10k,
    log1p_cp10k,
    observed_to_evaluator_counts,
    raw_output_to_count_equivalent,
    run_observed,
    verify_pinned_source,
)
from .registry import (
    MethodPlanEntry,
    MethodRegistry,
    load_method_registry,
    verify_cached_method_sources,
)
from .saver import (
    SAVERConfig,
    finalize_saver_direct_output,
    finalize_saver_output,
    run_saver,
    run_saver_direct,
    saver_to_evaluator_counts,
)
from .sccr import (
    SCCRConfig,
    finalize_sccr_direct_output,
    finalize_sccr_output,
    run_sccr,
    run_sccr_direct,
    sccr_to_evaluator_counts,
)
from .scsdae import (
    SCSDaeAttemptReceipt,
    SCSDaeConfig,
    SCSDaeUnavailableError,
    finalize_scsdae_direct_output,
    finalize_scsdae_output,
    run_scsdae,
    run_scsdae_direct,
    scsdae_to_evaluator_counts,
)
from .sctsi import (
    SCTSIAttemptReceipt,
    SCTSIConfig,
    SCTSIMatchedBulkReference,
    SCTSIUnavailableError,
    finalize_sctsi_output,
    prepare_sctsi_matched_bulk_reference,
    run_sctsi,
    sctsi_to_evaluator_counts,
    validate_sctsi_matched_bulk_reference,
)
from .scziva import (
    SCZivaConfig,
    finalize_scziva_direct_output,
    finalize_scziva_output,
    run_scziva,
    run_scziva_direct,
    scziva_to_evaluator_counts,
)
from .scvi import (
    SCVIConfig,
    finalize_scvi_direct_output,
    finalize_scvi_output,
    frequencies_to_observed_library_counts,
    run_scvi,
    run_scvi_direct,
    scvi_to_evaluator_counts,
)


def maskimpute_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Convert an in-tree raw output without importing the optional Torch stack."""

    return raw_output_to_count_equivalent(method_input, native_output)


CORE_EVALUATOR_COUNT_CONVERTERS = MappingProxyType(
    {
        "observed": observed_to_evaluator_counts,
        "capacity-matched-ae": maskimpute_to_evaluator_counts,
        "maskimpute": maskimpute_to_evaluator_counts,
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
        "capacity-matched-ae": "raw_counts",
        "maskimpute": "raw_counts",
        "alra": "log1p_cp10k",
        "magic": "log1p_cp10k",
        "dca": "raw_counts",
        "scvi": "raw_counts",
        "saver": "method_native_normalized",
    }
)
RECENT_EVALUATOR_COUNT_CONVERTERS = MappingProxyType(
    {
        "scziva": scziva_to_evaluator_counts,
        "afmf": afmf_to_evaluator_counts,
        "biaeimpute": biaeimpute_to_evaluator_counts,
        "d3impute": d3impute_to_evaluator_counts,
    }
)
RECENT_EVALUATOR_NATIVE_SCALES = MappingProxyType(
    {
        "scziva": "raw_counts",
        "afmf": "method_native_normalized",
        "biaeimpute": "raw_counts",
        "d3impute": "external_reference_adjusted",
    }
)
LEGACY_EVALUATOR_COUNT_CONVERTERS = MappingProxyType(
    {
        "sccr": sccr_to_evaluator_counts,
        "scsdae": scsdae_to_evaluator_counts,
    }
)
LEGACY_EVALUATOR_NATIVE_SCALES = MappingProxyType(
    {
        "sccr": "method_native_normalized",
        "scsdae": "method_native_normalized",
    }
)
EXTERNAL_REFERENCE_EVALUATOR_COUNT_CONVERTERS = MappingProxyType(
    {
        "d3impute": d3impute_to_evaluator_counts,
        "sctsi": sctsi_to_evaluator_counts,
    }
)
EXTERNAL_REFERENCE_EVALUATOR_NATIVE_SCALES = MappingProxyType(
    {
        "d3impute": "external_reference_adjusted",
        "sctsi": "external_reference_adjusted",
    }
)


def _output_to_evaluator_counts(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
    converters: Mapping[str, Callable[[MethodInput, object], np.ndarray]],
    native_scales: Mapping[str, str],
) -> np.ndarray:
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if not isinstance(snapshot, MethodOutputSnapshot):
        raise TypeError("snapshot must be a MethodOutputSnapshot")
    try:
        converter = converters[snapshot.method_id]
        expected_scale = native_scales[snapshot.method_id]
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


def core_output_to_evaluator_counts(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Validate a bound native snapshot and apply its declared count conversion."""

    return _output_to_evaluator_counts(
        method_input,
        snapshot,
        CORE_EVALUATOR_COUNT_CONVERTERS,
        CORE_EVALUATOR_NATIVE_SCALES,
    )


def core_output_to_evaluator_log2_cp10k(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Convert one declared core output to the common log2(CP10k+1) scale."""

    counts = core_output_to_evaluator_counts(method_input, snapshot)
    return count_equivalent_to_log2_cp10k(counts)


def recent_output_to_evaluator_counts(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Validate a recent-comparator snapshot and apply its declared conversion."""

    return _output_to_evaluator_counts(
        method_input,
        snapshot,
        RECENT_EVALUATOR_COUNT_CONVERTERS,
        RECENT_EVALUATOR_NATIVE_SCALES,
    )


def recent_output_to_evaluator_log2_cp10k(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Convert one recent comparator to the common log2(CP10k+1) scale."""

    counts = recent_output_to_evaluator_counts(method_input, snapshot)
    return count_equivalent_to_log2_cp10k(counts)


def legacy_output_to_evaluator_counts(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Validate a required legacy snapshot and apply its scale conversion."""

    return _output_to_evaluator_counts(
        method_input,
        snapshot,
        LEGACY_EVALUATOR_COUNT_CONVERTERS,
        LEGACY_EVALUATOR_NATIVE_SCALES,
    )


def legacy_output_to_evaluator_log2_cp10k(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Convert one required legacy comparator to common log2(CP10k+1)."""

    counts = legacy_output_to_evaluator_counts(method_input, snapshot)
    return count_equivalent_to_log2_cp10k(counts)


def external_reference_output_to_evaluator_counts(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Validate an external-reference snapshot and convert to evaluator counts."""

    return _output_to_evaluator_counts(
        method_input,
        snapshot,
        EXTERNAL_REFERENCE_EVALUATOR_COUNT_CONVERTERS,
        EXTERNAL_REFERENCE_EVALUATOR_NATIVE_SCALES,
    )


def external_reference_output_to_evaluator_log2_cp10k(
    method_input: MethodInput,
    snapshot: MethodOutputSnapshot,
) -> np.ndarray:
    """Convert an external-reference comparator to common log2(CP10k+1)."""

    counts = external_reference_output_to_evaluator_counts(method_input, snapshot)
    return count_equivalent_to_log2_cp10k(counts)


_LAZY_MASKIMPUTE_EXPORTS = frozenset(
    {
        "DirectMaskImputeExecution",
        "MaskImputeAdapterExecution",
        "finalize_maskimpute_output",
        "run_capacity_matched_ae",
        "run_frozen_final_in_tree",
        "run_maskimpute",
        "run_revision_maskimpute_direct",
    }
)


def __getattr__(name: str):
    if name in _LAZY_MASKIMPUTE_EXPORTS:
        from importlib import import_module

        value = getattr(import_module(".maskimpute", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AFMFConfig",
    "ALRAConfig",
    "AdapterExecution",
    "AdapterUnavailableError",
    "CitationSpec",
    "CompatibilityEvent",
    "CORE_EVALUATOR_COUNT_CONVERTERS",
    "CORE_EVALUATOR_NATIVE_SCALES",
    "CovariateColumn",
    "BiAEImputeConfig",
    "DCAConfig",
    "D3ImputeConfig",
    "DirectAdapterExecution",
    "DirectMaskImputeExecution",
    "DirectMethodOutput",
    "EnvironmentSpec",
    "EXTERNAL_REFERENCE_EVALUATOR_COUNT_CONVERTERS",
    "EXTERNAL_REFERENCE_EVALUATOR_NATIVE_SCALES",
    "LicenseSpec",
    "LEGACY_EVALUATOR_COUNT_CONVERTERS",
    "LEGACY_EVALUATOR_NATIVE_SCALES",
    "MethodContractError",
    "MethodInput",
    "MethodOutputSnapshot",
    "MethodPlanEntry",
    "MethodRegistry",
    "MethodRunRecord",
    "MethodSpec",
    "MethodStatusRow",
    "MAGICConfig",
    "MaskImputeAdapterExecution",
    "MatchedBulkReference",
    "RECENT_EVALUATOR_COUNT_CONVERTERS",
    "RECENT_EVALUATOR_NATIVE_SCALES",
    "ResourceSpec",
    "SAVERConfig",
    "SCCRConfig",
    "SCSDaeAttemptReceipt",
    "SCSDaeConfig",
    "SCSDaeUnavailableError",
    "SCTSIAttemptReceipt",
    "SCTSIConfig",
    "SCTSIMatchedBulkReference",
    "SCTSIUnavailableError",
    "SCVIConfig",
    "SCZivaConfig",
    "SourceSpec",
    "SourceReceipt",
    "build_method_status_table",
    "canonical_run_record_bytes",
    "afmf_to_evaluator_counts",
    "biaeimpute_to_evaluator_counts",
    "core_output_to_evaluator_counts",
    "core_output_to_evaluator_log2_cp10k",
    "count_equivalent_to_log2_cp10k",
    "d3impute_to_evaluator_counts",
    "dca_to_evaluator_counts",
    "alra_to_evaluator_counts",
    "finalize_alra_output",
    "finalize_alra_direct_output",
    "finalize_afmf_output",
    "finalize_afmf_direct_output",
    "finalize_biaeimpute_output",
    "finalize_biaeimpute_direct_output",
    "finalize_d3impute_output",
    "finalize_dca_output",
    "finalize_dca_direct_output",
    "finalize_direct_method_output",
    "finalize_magic_output",
    "finalize_magic_direct_output",
    "finalize_maskimpute_output",
    "finalize_saver_output",
    "finalize_saver_direct_output",
    "finalize_sccr_output",
    "finalize_sccr_direct_output",
    "finalize_scsdae_output",
    "finalize_scsdae_direct_output",
    "finalize_sctsi_output",
    "finalize_scvi_output",
    "finalize_scvi_direct_output",
    "finalize_scziva_output",
    "finalize_scziva_direct_output",
    "frequencies_to_observed_library_counts",
    "external_reference_output_to_evaluator_counts",
    "external_reference_output_to_evaluator_log2_cp10k",
    "load_method_registry",
    "legacy_output_to_evaluator_counts",
    "legacy_output_to_evaluator_log2_cp10k",
    "log1p_cp10k",
    "magic_to_evaluator_counts",
    "maskimpute_to_evaluator_counts",
    "observed_to_evaluator_counts",
    "prepare_method_input",
    "prepare_matched_bulk_reference",
    "prepare_sctsi_matched_bulk_reference",
    "recent_output_to_evaluator_counts",
    "recent_output_to_evaluator_log2_cp10k",
    "snapshot_method_output",
    "run_alra",
    "run_alra_direct",
    "run_afmf",
    "run_afmf_direct",
    "run_biaeimpute",
    "run_biaeimpute_direct",
    "run_d3impute",
    "run_dca",
    "run_dca_direct",
    "run_magic",
    "run_magic_direct",
    "run_capacity_matched_ae",
    "run_frozen_final_in_tree",
    "run_maskimpute",
    "run_revision_maskimpute_direct",
    "run_observed",
    "run_saver",
    "run_saver_direct",
    "run_sccr",
    "run_sccr_direct",
    "run_scsdae",
    "run_scsdae_direct",
    "run_sctsi",
    "run_scvi",
    "run_scvi_direct",
    "run_scziva",
    "run_scziva_direct",
    "saver_to_evaluator_counts",
    "sccr_to_evaluator_counts",
    "scsdae_to_evaluator_counts",
    "sctsi_to_evaluator_counts",
    "scvi_to_evaluator_counts",
    "scziva_to_evaluator_counts",
    "validate_matched_bulk_reference",
    "validate_sctsi_matched_bulk_reference",
    "validate_run_record",
    "verify_pinned_source",
    "verify_cached_method_sources",
]
