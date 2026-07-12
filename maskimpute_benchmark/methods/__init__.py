"""Strict method registry and leakage-safe benchmark execution contracts."""

from .base import (
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
from .registry import (
    MethodRegistry,
    load_method_registry,
    verify_cached_method_sources,
)

__all__ = [
    "CitationSpec",
    "CovariateColumn",
    "EnvironmentSpec",
    "LicenseSpec",
    "MethodContractError",
    "MethodInput",
    "MethodOutputSnapshot",
    "MethodRegistry",
    "MethodRunRecord",
    "MethodSpec",
    "MethodStatusRow",
    "ResourceSpec",
    "SourceSpec",
    "build_method_status_table",
    "canonical_run_record_bytes",
    "load_method_registry",
    "prepare_method_input",
    "snapshot_method_output",
    "validate_run_record",
    "verify_cached_method_sources",
]
