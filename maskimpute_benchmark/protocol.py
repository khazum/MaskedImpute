"""Canonical loading and hashing for the publication study protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FINAL_MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")


@dataclass(frozen=True, slots=True)
class DevelopmentProtocol:
    """Development-suite dimensions and artifact namespace."""

    namespace: str
    draws_per_condition: int
    cells: int
    genes: int


@dataclass(frozen=True, slots=True)
class FinalProtocol:
    """Final-suite dimensions, mechanisms, and artifact namespace."""

    namespace: str
    mechanisms: tuple[str, ...]
    draws_per_condition: int
    model_seeds: int
    cells: int
    genes: int


@dataclass(frozen=True, slots=True)
class Protocol:
    """Validated, immutable publication study protocol."""

    schema_version: int
    legacy_data_role: str
    development: DevelopmentProtocol
    final: FinalProtocol
    primary_metrics: tuple[str, ...]
    final_timeout_seconds: int
    max_rss_gib: int | float
    max_gpu_gib: int | float

    @property
    def mechanisms(self) -> tuple[str, ...]:
        return self.final.mechanisms

    @property
    def final_draws_per_condition(self) -> int:
        return self.final.draws_per_condition

    @property
    def final_model_seeds(self) -> int:
        return self.final.model_seeds


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of the canonical JSON representation of *value*."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the bytes stored at *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: object, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return value


def _namespace(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_protocol(path: Path) -> Protocol:
    """Load and validate schema version 1 of the publication protocol."""

    data = _mapping(
        json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        ),
        "protocol",
    )

    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be 1")

    final_data = _mapping(data.get("final"), "final")
    mechanisms_value = final_data.get("mechanisms")
    if isinstance(mechanisms_value, list) and any(
        isinstance(mechanism, str) and mechanism.casefold() == "splatter"
        for mechanism in mechanisms_value
    ):
        raise ValueError("Splatter is development-only and cannot be a final mechanism")
    if not isinstance(mechanisms_value, list) or not all(
        isinstance(mechanism, str) for mechanism in mechanisms_value
    ):
        raise ValueError("final.mechanisms must be a list of strings")
    mechanisms = tuple(mechanisms_value)
    if mechanisms != FINAL_MECHANISMS:
        raise ValueError(f"final.mechanisms must be {FINAL_MECHANISMS!r} in that order")

    development_data = _mapping(data.get("development"), "development")
    development_namespace = _namespace(
        development_data.get("namespace"), "development.namespace"
    )
    final_namespace = _namespace(final_data.get("namespace"), "final.namespace")
    if development_namespace == final_namespace:
        raise ValueError("development and final namespaces must be disjoint")

    legacy_data_role = data.get("legacy_data_role")
    if legacy_data_role != "development_only":
        raise ValueError("legacy_data_role must be development_only")

    primary_metrics_value = data.get("primary_metrics")
    if (
        not isinstance(primary_metrics_value, list)
        or not primary_metrics_value
        or not all(
            isinstance(metric, str) and metric for metric in primary_metrics_value
        )
    ):
        raise ValueError("primary_metrics must be a nonempty list of strings")

    development = DevelopmentProtocol(
        namespace=development_namespace,
        draws_per_condition=_positive_int(
            development_data.get("draws_per_condition"),
            "development.draws_per_condition",
        ),
        cells=_positive_int(development_data.get("cells"), "development.cells"),
        genes=_positive_int(development_data.get("genes"), "development.genes"),
    )
    final = FinalProtocol(
        namespace=final_namespace,
        mechanisms=mechanisms,
        draws_per_condition=_positive_int(
            final_data.get("draws_per_condition"), "final.draws_per_condition"
        ),
        model_seeds=_positive_int(final_data.get("model_seeds"), "final.model_seeds"),
        cells=_positive_int(final_data.get("cells"), "final.cells"),
        genes=_positive_int(final_data.get("genes"), "final.genes"),
    )

    return Protocol(
        schema_version=1,
        legacy_data_role=legacy_data_role,
        development=development,
        final=final,
        primary_metrics=tuple(primary_metrics_value),
        final_timeout_seconds=_positive_int(
            data.get("final_timeout_seconds"), "final_timeout_seconds"
        ),
        max_rss_gib=_positive_number(data.get("max_rss_gib"), "max_rss_gib"),
        max_gpu_gib=_positive_number(data.get("max_gpu_gib"), "max_gpu_gib"),
    )
