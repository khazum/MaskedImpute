"""Strict loading and source verification for the publication method registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from types import MappingProxyType
from typing import Any

from .base import (
    CitationSpec,
    EnvironmentSpec,
    LicenseSpec,
    MethodContractError,
    MethodSpec,
    ResourceSpec,
    SourceSpec,
)


_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SAFE_REASON = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DOI = re.compile(r"10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+\Z")
_DECLARED_SPDX = frozenset({"Apache-2.0", "BSD-3-Clause", "GPL-2.0-only", "MIT"})
_SCALES = frozenset(
    {
        "raw_counts",
        "log1p_cp10k",
        "method_native_normalized",
        "external_reference_adjusted",
    }
)
_SOURCE_POLICIES = frozenset(
    {
        "study_freeze_bound_in_tree",
        "pinned_adapter_isolated_environment",
        "invoke_pristine_source_no_redistribution",
    }
)
_INTEGRATION_STATUSES = frozenset(
    {"implemented", "pending", "pending_legacy_attempt", "unavailable"}
)
_EXECUTION_SCOPES = frozenset(
    {
        "same_input_required",
        "external_reference_only",
        "historical_not_run",
        "not_applicable",
    }
)
_METHOD_KEYS = frozenset(
    {
        "id",
        "display_name",
        "role",
        "track",
        "execution_scope",
        "applicability_reason",
        "input_scale",
        "output_scale",
        "stochastic",
        "seed_policy",
        "source",
        "license",
        "citation",
        "environment",
        "resources",
        "preserves_observed_positives",
        "source_policy",
        "integration_status",
        "integration_reason",
    }
)


@dataclass(frozen=True, slots=True)
class MethodPlanEntry:
    """Minimal immutable scheduling decision derived from the method registry."""

    method_id: str
    execution_scope: str
    applicability_reason: str | None
    executable: bool


@dataclass(frozen=True, slots=True)
class MethodRegistry:
    """Immutable ordered method denominator."""

    schema_version: int
    methods: tuple[MethodSpec, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(spec.id for spec in self.methods)

    def by_id(self, method_id: str) -> MethodSpec:
        for spec in self.methods:
            if spec.id == method_id:
                return spec
        raise KeyError(method_id)

    def execution_plan(self) -> tuple[MethodPlanEntry, ...]:
        """Return the closed scheduling/applicability plan in registry order."""

        return tuple(
            MethodPlanEntry(
                method_id=spec.id,
                execution_scope=spec.execution_scope,
                applicability_reason=spec.applicability_reason,
                executable=spec.executable,
            )
            for spec in self.methods
        )


def _reject_constant(value: str) -> None:
    raise MethodContractError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MethodContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: object, name: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MethodContractError(f"{name} must be a JSON object")
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise MethodContractError(
            f"{name} fields are not closed; missing={missing}, extra={extra}"
        )
    return value


def _string(value: object, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MethodContractError(f"{name} must be a nonempty string")
    return value


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise MethodContractError(
            f"{name} must be a safe lowercase hyphen-separated identifier"
        )
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise MethodContractError(f"{name} must be a positive integer")
    return value


def _resource_number(
    value: object,
    name: str,
    *,
    allow_zero: bool,
) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise MethodContractError(f"{name} must be finite and {qualifier}")
    return value


def _parse_source(value: object, method_id: str) -> SourceSpec:
    data = _mapping(
        value,
        f"method {method_id} source",
        frozenset({"kind", "url", "revision", "tree", "cache_path", "freeze_binding"}),
    )
    kind = data["kind"]
    if kind == "git":
        url = _string(data["url"], f"method {method_id} source url")
        if not url.startswith("https://") or not url.endswith(".git"):
            raise MethodContractError(
                f"method {method_id} source url must be an HTTPS git URL"
            )
        revision = data["revision"]
        if not isinstance(revision, str) or not _GIT_OBJECT.fullmatch(revision):
            raise MethodContractError(
                f"method {method_id} source revision must be a 40-character git object"
            )
        tree = data["tree"]
        if not isinstance(tree, str) or not _GIT_OBJECT.fullmatch(tree):
            raise MethodContractError(
                f"method {method_id} source tree must be a 40-character git object"
            )
        cache_path = _string(
            data["cache_path"], f"method {method_id} source cache_path"
        )
        pure_path = PurePosixPath(cache_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            raise MethodContractError(
                f"method {method_id} source cache_path must be a safe relative path"
            )
        if data["freeze_binding"] is not None:
            raise MethodContractError(
                f"method {method_id} git source freeze_binding must be null"
            )
        return SourceSpec(kind, url, revision, tree, cache_path, None)
    if kind == "in_tree":
        for field in ("url", "revision", "tree", "cache_path"):
            if data[field] is not None:
                raise MethodContractError(
                    f"method {method_id} in-tree source {field} must be null"
                )
        if data["freeze_binding"] != "study_freeze_commit":
            raise MethodContractError(
                f"method {method_id} in-tree source requires study_freeze_commit binding"
            )
        return SourceSpec("in_tree", None, None, None, None, "study_freeze_commit")
    raise MethodContractError(f"method {method_id} source kind must be git or in_tree")


def _parse_license(value: object, method_id: str) -> LicenseSpec:
    data = _mapping(
        value,
        f"method {method_id} license",
        frozenset({"status", "spdx", "notice"}),
    )
    status = data["status"]
    if status not in {"declared", "NOASSERTION", "pending"}:
        raise MethodContractError(
            f"method {method_id} license status must be declared, NOASSERTION, or pending"
        )
    notice = _string(
        data["notice"], f"method {method_id} license notice", nullable=True
    )
    if status == "declared":
        spdx = data["spdx"]
        if spdx not in _DECLARED_SPDX:
            raise MethodContractError(
                f"method {method_id} declared license requires a verified SPDX identifier"
            )
    else:
        if data["spdx"] is not None:
            raise MethodContractError(
                f"method {method_id} {status} license spdx must be null"
            )
        spdx = None
    return LicenseSpec(status=status, spdx=spdx, notice=notice)


def _parse_citation(value: object, method_id: str) -> CitationSpec:
    data = _mapping(
        value,
        f"method {method_id} citation",
        frozenset({"status", "doi", "url"}),
    )
    status = data["status"]
    if status not in {"verified", "pending"}:
        raise MethodContractError(
            f"method {method_id} citation status must be verified or pending"
        )
    url = _string(data["url"], f"method {method_id} citation url", nullable=True)
    if url is not None and not url.startswith("https://"):
        raise MethodContractError(f"method {method_id} citation url must use HTTPS")
    doi = data["doi"]
    if status == "verified":
        if not isinstance(doi, str) or not _DOI.fullmatch(doi):
            raise MethodContractError(
                f"method {method_id} verified citation requires a valid DOI"
            )
    elif doi is not None:
        raise MethodContractError(
            f"method {method_id} pending citation DOI must be null"
        )
    return CitationSpec(status=status, doi=doi, url=url)


def _parse_environment(value: object, method_id: str) -> EnvironmentSpec:
    data = _mapping(
        value,
        f"method {method_id} environment",
        frozenset({"id", "status", "lock_sha256"}),
    )
    environment_id = _safe_id(data["id"], f"method {method_id} environment id")
    status = data["status"]
    if status not in {"pending", "ready", "failed"}:
        raise MethodContractError(
            f"method {method_id} environment status must be pending, ready, or failed"
        )
    lock_sha256 = data["lock_sha256"]
    if status == "ready":
        if not isinstance(lock_sha256, str) or not _SHA256.fullmatch(lock_sha256):
            raise MethodContractError(
                f"method {method_id} ready environment requires lock_sha256"
            )
    elif lock_sha256 is not None:
        raise MethodContractError(
            f"method {method_id} non-ready environment lock_sha256 must be null"
        )
    return EnvironmentSpec(environment_id, status, lock_sha256)


def _parse_resources(value: object, method_id: str) -> ResourceSpec:
    data = _mapping(
        value,
        f"method {method_id} resources",
        frozenset(
            {
                "timeout_seconds",
                "cpu_cores",
                "gpu_required",
                "max_rss_gib",
                "max_gpu_gib",
            }
        ),
    )
    timeout = _positive_int(
        data["timeout_seconds"], f"method {method_id} resources timeout_seconds"
    )
    cpu_cores = _positive_int(
        data["cpu_cores"], f"method {method_id} resources cpu_cores"
    )
    gpu_required = data["gpu_required"]
    if type(gpu_required) is not bool:
        raise MethodContractError(
            f"method {method_id} resources gpu_required must be boolean"
        )
    max_rss = _resource_number(
        data["max_rss_gib"],
        f"method {method_id} resources max_rss_gib",
        allow_zero=False,
    )
    max_gpu = _resource_number(
        data["max_gpu_gib"],
        f"method {method_id} resources max_gpu_gib",
        allow_zero=True,
    )
    if gpu_required and max_gpu == 0:
        raise MethodContractError(
            f"method {method_id} GPU method max_gpu_gib must be positive"
        )
    if not gpu_required and max_gpu != 0:
        raise MethodContractError(
            f"method {method_id} CPU-only method max_gpu_gib must be zero"
        )
    return ResourceSpec(timeout, cpu_cores, gpu_required, max_rss, max_gpu)


def _parse_method(value: object) -> MethodSpec:
    data = _mapping(value, "method", _METHOD_KEYS)
    method_id = _safe_id(data["id"], "method id")
    display_name = _string(data["display_name"], f"method {method_id} display_name")
    role = data["role"]
    if role not in {"control", "candidate", "competitor"}:
        raise MethodContractError(
            f"method {method_id} role must be control, candidate, or competitor"
        )
    track = data["track"]
    if track not in {"same_input", "external_reference"}:
        raise MethodContractError(
            f"method {method_id} track must be same_input or external_reference"
        )
    execution_scope = data["execution_scope"]
    if execution_scope not in _EXECUTION_SCOPES:
        raise MethodContractError(
            f"method {method_id} execution_scope must be one of {sorted(_EXECUTION_SCOPES)}"
        )
    applicability_reason = _string(
        data["applicability_reason"],
        f"method {method_id} applicability_reason",
        nullable=True,
    )
    if applicability_reason is not None and not _SAFE_REASON.fullmatch(
        applicability_reason
    ):
        raise MethodContractError(
            f"method {method_id} applicability_reason must be a safe reason code"
        )
    if execution_scope == "same_input_required":
        if track != "same_input":
            raise MethodContractError(
                f"method {method_id} same_input_required scope requires same_input track"
            )
        if applicability_reason is not None:
            raise MethodContractError(
                f"method {method_id} executable scope applicability_reason must be null"
            )
    elif execution_scope == "external_reference_only":
        if track != "external_reference":
            raise MethodContractError(
                f"method {method_id} external_reference_only scope requires external_reference track"
            )
        if applicability_reason is not None:
            raise MethodContractError(
                f"method {method_id} executable scope applicability_reason must be null"
            )
    elif execution_scope == "historical_not_run":
        if track != "same_input":
            raise MethodContractError(
                f"method {method_id} historical_not_run scope requires same_input track"
            )
        if applicability_reason is not None:
            raise MethodContractError(
                f"method {method_id} historical_not_run applicability_reason must be null"
            )
    elif applicability_reason is None:
        raise MethodContractError(
            f"method {method_id} not_applicable scope requires applicability_reason"
        )
    for field in ("input_scale", "output_scale"):
        if data[field] not in _SCALES:
            raise MethodContractError(
                f"method {method_id} {field} must be one of {sorted(_SCALES)}"
            )
    stochastic = data["stochastic"]
    if type(stochastic) is not bool:
        raise MethodContractError(f"method {method_id} stochastic must be boolean")
    seed_policy = data["seed_policy"]
    expected_seed_policy = "required" if stochastic else "not_applicable"
    if seed_policy != expected_seed_policy:
        raise MethodContractError(
            f"method {method_id} seed_policy must be {expected_seed_policy}"
        )
    source = _parse_source(data["source"], method_id)
    license_spec = _parse_license(data["license"], method_id)
    citation = _parse_citation(data["citation"], method_id)
    environment = _parse_environment(data["environment"], method_id)
    resources = _parse_resources(data["resources"], method_id)
    preserves = data["preserves_observed_positives"]
    if type(preserves) is not bool:
        raise MethodContractError(
            f"method {method_id} preserves_observed_positives must be boolean"
        )
    if preserves and (
        data["input_scale"] != "raw_counts" or data["output_scale"] != "raw_counts"
    ):
        raise MethodContractError(
            f"method {method_id} positive preservation requires raw_counts I/O"
        )
    source_policy = data["source_policy"]
    if source_policy not in _SOURCE_POLICIES:
        raise MethodContractError(f"method {method_id} has invalid source_policy")
    if (
        license_spec.status == "NOASSERTION"
        and source_policy != "invoke_pristine_source_no_redistribution"
    ):
        raise MethodContractError(
            f"method {method_id} NOASSERTION source must use pristine no-redistribution policy"
        )
    if source.kind == "in_tree" and source_policy != "study_freeze_bound_in_tree":
        raise MethodContractError(
            f"method {method_id} in-tree source must use freeze-bound policy"
        )
    integration_status = data["integration_status"]
    if integration_status not in _INTEGRATION_STATUSES:
        raise MethodContractError(f"method {method_id} has invalid integration_status")
    integration_reason = _string(
        data["integration_reason"],
        f"method {method_id} integration_reason",
        nullable=True,
    )
    if integration_status == "unavailable" and integration_reason is None:
        raise MethodContractError(
            f"method {method_id} unavailable integration requires a reason"
        )
    return MethodSpec(
        id=method_id,
        display_name=display_name,
        role=role,
        track=track,
        execution_scope=execution_scope,
        applicability_reason=applicability_reason,
        input_scale=data["input_scale"],
        output_scale=data["output_scale"],
        stochastic=stochastic,
        seed_policy=seed_policy,
        source=source,
        license=license_spec,
        citation=citation,
        environment=environment,
        resources=resources,
        preserves_observed_positives=preserves,
        source_policy=source_policy,
        integration_status=integration_status,
        integration_reason=integration_reason,
    )


def load_method_registry(path: Path) -> MethodRegistry:
    """Load the strict version-1 method denominator from canonical JSON."""

    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except OSError as error:
        raise MethodContractError(f"could not read method registry: {path}") from error
    except json.JSONDecodeError as error:
        raise MethodContractError(
            f"method registry is not valid JSON: {error}"
        ) from error
    root = _mapping(data, "method registry", frozenset({"schema_version", "methods"}))
    if root["schema_version"] != 1 or type(root["schema_version"]) is not int:
        raise MethodContractError("method registry schema_version must be 1")
    method_values = root["methods"]
    if not isinstance(method_values, list) or not method_values:
        raise MethodContractError("method registry methods must be a nonempty list")
    methods = tuple(_parse_method(value) for value in method_values)
    ids = [spec.id for spec in methods]
    duplicates = sorted({method_id for method_id in ids if ids.count(method_id) > 1})
    if duplicates:
        raise MethodContractError(f"duplicate method id: {', '.join(duplicates)}")
    return MethodRegistry(schema_version=1, methods=methods)


def _git(path: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise MethodContractError(
            f"could not inspect cached git source: {path}"
        ) from error
    return result.stdout.strip()


def verify_cached_method_sources(
    registry: MethodRegistry,
    *,
    repository_root: Path,
    require_all: bool = False,
) -> Mapping[str, str]:
    """Read-only verify available upstream checkouts against every declared pin."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a pathlib.Path")
    statuses: dict[str, str] = {}
    for spec in registry.methods:
        source = spec.source
        if source.kind != "git":
            continue
        assert source.cache_path is not None
        path = repository_root / source.cache_path
        if not path.is_dir():
            if require_all:
                raise MethodContractError(
                    f"cached source is missing for method {spec.id}: {source.cache_path}"
                )
            statuses[spec.id] = "missing"
            continue
        observed_revision = _git(path, "rev-parse", "HEAD")
        observed_tree = _git(path, "rev-parse", "HEAD^{tree}")
        observed_url = _git(path, "remote", "get-url", "origin")
        observed_status = _git(
            path, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if observed_revision != source.revision:
            raise MethodContractError(
                f"cached source revision mismatch for method {spec.id}"
            )
        if observed_tree != source.tree:
            raise MethodContractError(
                f"cached source tree mismatch for method {spec.id}"
            )
        if observed_url != source.url:
            raise MethodContractError(
                f"cached source URL mismatch for method {spec.id}"
            )
        if observed_status:
            raise MethodContractError(
                f"cached source is not pristine for method {spec.id}"
            )
        statuses[spec.id] = "verified"
    return MappingProxyType(statuses)


__all__ = [
    "MethodPlanEntry",
    "MethodRegistry",
    "load_method_registry",
    "verify_cached_method_sources",
]
