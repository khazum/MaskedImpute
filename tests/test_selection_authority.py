from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import replace
import inspect
import hashlib
import json
import os
from pathlib import Path
import shutil
import importlib.util
import subprocess
import sys
from types import SimpleNamespace
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_FILES = (
    "study/protocol.json",
    "study/development_panel.json",
    "study/methods.json",
    "study/ablations.json",
    "study/calibration_contract.json",
    "study/comparator_tuning.json",
    "study/selection_contract.json",
    "study/development_search.json",
)

FORBIDDEN_DIRECT_IDENTITY_TOKENS = (
    "hash",
    "digest",
    "checksum",
    "fingerprint",
    "sha",
)
FORBIDDEN_DIRECT_HELPERS = {
    "canonical_sha256",
    "method_input_sha256",
    "implementation_source_sha256",
    "_file_sha256",
    "_stable_file_sha256",
    "_digest",
    "sha256",
}


def _direct_schema_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for nested in value.values() for key in _direct_schema_keys(nested)
        )
    if isinstance(value, list):
        return tuple(key for nested in value for key in _direct_schema_keys(nested))
    return ()


def _forbidden_direct_key(name: str) -> bool:
    lowered = name.casefold()
    return lowered != "shape" and any(
        token in lowered for token in FORBIDDEN_DIRECT_IDENTITY_TOKENS
    )


def _module_symbol_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                local = imported.asname or imported.name.split(".")[0]
                aliases[local] = imported.name
        elif isinstance(node, ast.ImportFrom):
            prefix = "" if node.module is None else f"{node.module}."
            for imported in node.names:
                local = imported.asname or imported.name
                aliases[local] = f"{prefix}{imported.name}"
    return aliases


def _scope_import_aliases(scope: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local = imported.asname or imported.name.split(".")[0]
                aliases[local] = imported.name
        elif isinstance(node, ast.ImportFrom):
            prefix = "" if node.module is None else f"{node.module}."
            for imported in node.names:
                local = imported.asname or imported.name
                aliases[local] = f"{prefix}{imported.name}"
    return aliases


def _resolve_symbol(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolve_symbol(node.value, aliases)
        return None if owner is None else f"{owner}.{node.attr}"
    return None


def _literal_string(node: ast.AST, strings: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return strings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left, strings)
        right = _literal_string(node.right, strings)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        values: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                resolved = _literal_string(value.value, strings)
            else:
                resolved = _literal_string(value, strings)
            if resolved is None:
                return None
            values.append(resolved)
        return "".join(values)
    return None


def _scope_aliases(
    scope: ast.AST,
    module_aliases: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    symbols = dict(module_aliases)
    symbols.update(_scope_import_aliases(scope))
    strings: dict[str, str] = {}
    assignments = tuple(
        node
        for node in ast.walk(scope)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    )
    for _pass in range(len(assignments) + 1):
        changed = False
        for assignment in assignments:
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else (assignment.target,)
            )
            value = assignment.value
            if value is None:
                continue
            symbol = _resolve_symbol(value, symbols)
            string = _literal_string(value, strings)
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if symbol is not None and symbols.get(target.id) != symbol:
                    symbols[target.id] = symbol
                    changed = True
                if string is not None and strings.get(target.id) != string:
                    strings[target.id] = string
                    changed = True
        if not changed:
            break
    return symbols, strings


def _audited_direct_scope(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        "direct" in node.name.casefold().split("_")
        or "fair_comparator" in node.name.casefold()
    )


def _direct_source_audit_findings(
    source: str,
    *,
    shared: bool,
) -> tuple[str, ...]:
    tree = ast.parse(source)
    module_aliases = _module_symbol_aliases(tree)
    scopes: tuple[ast.AST, ...] = (
        tuple(node for node in ast.walk(tree) if _audited_direct_scope(node))
        if shared
        else (tree,)
    )
    findings: set[str] = set()
    if not shared:
        for local, imported in module_aliases.items():
            leaf = imported.rsplit(".", 1)[-1]
            if imported == "hashlib" or leaf in FORBIDDEN_DIRECT_HELPERS:
                findings.add(f"forbidden import {local}={imported}")
    for scope in scopes:
        symbols, strings = _scope_aliases(scope, module_aliases)
        for local, imported in _scope_import_aliases(scope).items():
            leaf = imported.rsplit(".", 1)[-1]
            if imported == "hashlib" or leaf in FORBIDDEN_DIRECT_HELPERS:
                findings.add(f"forbidden import {local}={imported}")
        for call in (node for node in ast.walk(scope) if isinstance(node, ast.Call)):
            resolved = _resolve_symbol(call.func, symbols)
            if resolved is not None and (
                resolved == "hashlib"
                or resolved.startswith("hashlib.")
                or resolved.rsplit(".", 1)[-1] in FORBIDDEN_DIRECT_HELPERS
            ):
                findings.add(f"forbidden call {resolved}")
        for dictionary in (
            node for node in ast.walk(scope) if isinstance(node, ast.Dict)
        ):
            for key in dictionary.keys:
                if key is None:
                    continue
                resolved_key = _literal_string(key, strings)
                if resolved_key is not None and _forbidden_direct_key(resolved_key):
                    findings.add(f"forbidden generated key {resolved_key}")

    if not shared:
        symbols, _strings = _scope_aliases(tree, module_aliases)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            decorators = tuple(
                decorator.func if isinstance(decorator, ast.Call) else decorator
                for decorator in node.decorator_list
            )
            if not any(
                (resolved := _resolve_symbol(decorator, symbols)) is not None
                and resolved.rsplit(".", 1)[-1] == "dataclass"
                for decorator in decorators
            ):
                continue
            for child in node.body:
                if (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and _forbidden_direct_key(child.target.id)
                ):
                    findings.add(f"forbidden dataclass field {child.target.id}")
    return tuple(sorted(findings))


def _synthetic_direct_artifacts() -> dict[str, dict[str, object]]:
    import anndata as ad
    import numpy as np
    import pandas as pd

    from maskimpute_benchmark.comparator_tuning import (
        ComparatorAuthorityReference,
        comparator_method_binding,
        load_comparator_tuning_authority,
    )
    from maskimpute_benchmark.fair_comparator_checkpoint import (
        DirectCheckpointReport,
    )
    from maskimpute_benchmark.fair_comparator_execution import (
        DirectEvaluatedAttempt,
        DirectExecutionRequest,
        DirectLogReceipt,
        DirectPreZeroEvidence,
        DirectRunResult,
    )
    from maskimpute_benchmark.fair_comparator_plan import (
        ComparatorRunIdentity,
        DirectAuthorizedConfiguration,
        DirectCompetitionPlan,
        DirectPlanEntry,
        PreparedInputDescriptor,
        direct_run_id,
    )
    from maskimpute_benchmark.methods import (
        load_method_registry,
        prepare_method_input,
    )
    from maskimpute_benchmark.runner import SELECTION_COMPLETENESS_BLOCKERS
    from maskimpute_benchmark.selection import project_direct_selected_comparators

    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    row = authority.configurations_for("magic")[0]
    spec = registry.by_id("magic")
    method = comparator_method_binding(spec)

    def freeze(value: object) -> object:
        if isinstance(value, dict):
            return tuple((key, freeze(nested)) for key, nested in sorted(value.items()))
        if isinstance(value, list):
            return tuple(freeze(nested) for nested in value)
        return value

    payload = freeze(dict(row.payload))
    assert isinstance(payload, tuple)
    configuration = DirectAuthorizedConfiguration(
        method=method,
        configuration_id=row.configuration_id,
        configuration_kind="comparator_tuning",
        payload=payload,
        requires_count_score=False,
        requires_calibration=False,
    )
    identity = ComparatorRunIdentity(
        workflow_schema="maskimpute-fair-comparator-run-v1",
        authority_revision="fair-comparator-direct-v1",
        ordinal=1,
        method=method,
        configuration_id=row.configuration_id,
        configuration_kind="comparator_tuning",
        configuration_payload=payload,
        dataset_id="dataset-synthetic",
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        mask_seed=20_001,
        model_seed=42,
        draw_index=1,
    )
    entry = DirectPlanEntry(
        run_id=direct_run_id(identity),
        identity=identity,
        preflight_status="planned",
        preflight_reason=None,
        requires_count_score=False,
        requires_calibration=False,
    )
    descriptor = PreparedInputDescriptor(
        dataset_id="dataset-synthetic",
        source_reference="synthetic/in-memory",
        preprocessing_revision="paired-zero-library-union-v1",
        shape=(2, 2),
        dtype="<f8",
        cell_ids=("cell-1", "cell-2"),
        gene_ids=("gene-1", "gene-2"),
        batch_labels=(),
        total_count=3.0,
        nonzero_count=2,
        minimum=0.0,
        maximum=2.0,
        mechanism="symsim",
        mask_seed=20_001,
        technical_view="moderate",
    )
    plan = DirectCompetitionPlan(
        schema_version=1,
        identity_mode="direct-v1",
        authority_revision="fair-comparator-direct-v1",
        inputs=(descriptor,),
        entries=(entry,),
        configurations=(configuration,),
    )
    counts = np.asarray([[1, 0], [0, 2]], dtype=np.int64)
    view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=descriptor.cell_ids),
        var=pd.DataFrame(index=descriptor.gene_ids),
    )
    view.uns["source_dataset_sha256"] = "a" * 64
    view.uns["allowed_covariates"] = {"obs": [], "var": []}
    method_input = prepare_method_input(view)
    request = DirectExecutionRequest(
        identity=identity,
        method_spec=spec,
        method_input=method_input,
        timeout_seconds=1.0,
        max_rss_bytes=1,
        max_gpu_bytes=0,
    )
    reason = "synthetic_unavailable"
    attempt = DirectEvaluatedAttempt(
        run=DirectRunResult(
            run_id=entry.run_id,
            identity=identity,
            status="unavailable",
            reason=reason,
            runtime_seconds=1.0,
            peak_rss_bytes=1,
            peak_gpu_bytes=0,
            rss_measurement="synthetic_parent_rss",
            gpu_measurement="not_applicable_cpu",
            excluded_cell_count=0,
            excluded_cell_ids=(),
            retained_cell_count=2,
            retained_cell_ids=descriptor.cell_ids,
            retained_gene_count=2,
            observed_zero_count=2,
            stdout=DirectLogReceipt(
                stream="stdout",
                original_byte_count=0,
                capture_policy="discard_content",
                terminal_reason=reason,
            ),
            stderr=DirectLogReceipt(
                stream="stderr",
                original_byte_count=0,
                capture_policy="discard_content",
                terminal_reason=reason,
            ),
        ),
        metrics=(),
        native_output=None,
        native_output_scale=None,
        evaluator_output=None,
        p_pre_zero_evidence=DirectPreZeroEvidence(
            applicable=False,
            status="not_applicable",
            reason="method_does_not_emit_p_pre_zero",
            shape=None,
            dtype=None,
            encoding=None,
            path=None,
            compressed_byte_count=0,
        ),
    )
    checkpoint = DirectCheckpointReport(
        schema_version=1,
        identity_mode="direct-v1",
        authority_revision="fair-comparator-direct-v1",
        plan_snapshot=plan.to_dict(),
        input_descriptors=(descriptor,),
        planned_run_count=1,
        status="completed",
        evaluation_scope="reconstruction_only",
        comparator_selection_status="complete_terminal_denominator",
        selection_complete=False,
        selection_blockers=SELECTION_COMPLETENESS_BLOCKERS,
        records=(attempt.to_dict(),),
        budget={},
    )
    projection = project_direct_selected_comparators(
        ComparatorAuthorityReference(
            path="study/comparator_tuning.json",
            schema_version=2,
            authority_revision="fair-comparator-direct-v1",
        ),
        authority,
        (row,),
    )
    return {
        "request": request.to_dict(),
        "plan": plan.to_dict(),
        "attempt": attempt.to_dict(),
        "checkpoint": checkpoint.to_dict(),
        "projection": projection,
    }


@pytest.mark.parametrize(
    ("source", "shared"),
    (
        (
            """
from provenance import canonical_sha256

def project_direct_payload(value):
    summarize = canonical_sha256
    return summarize(value)
""",
            True,
        ),
        (
            """
import provenance as provenance_helpers

def project_direct_payload(value):
    return provenance_helpers.canonical_sha256(value)
""",
            True,
        ),
        (
            """
import dataclasses as dc

@dc.dataclass(frozen=True)
class DirectPlan:
    plan_sha256: str
""",
            False,
        ),
        (
            """
def project_direct_payload(value):
    key = "plan_" + "sha256"
    return {key: value}
""",
            True,
        ),
    ),
)
def test_direct_source_audit_rejects_alias_and_generated_key_evasions(
    source: str,
    shared: bool,
) -> None:
    assert _direct_source_audit_findings(source, shared=shared)


def test_direct_source_audit_leaves_unrelated_legacy_functions_out_of_scope() -> None:
    source = """
from provenance import canonical_sha256

def legacy_projection(value):
    alias = canonical_sha256
    return {"plan_sha256": alias(value)}

def project_direct_payload(value):
    return {"payload": value}
"""
    assert _direct_source_audit_findings(source, shared=True) == ()


@pytest.mark.parametrize(
    "source",
    (
        """
def project_direct_payload(value):
    import hashlib as local_hashlib

    return local_hashlib.new("sha256", value)
""",
        """
def project_direct_payload(value):
    from provenance import canonical_sha256 as summarize

    return summarize(value)
""",
    ),
)
def test_direct_source_audit_rejects_function_local_import_aliases(
    source: str,
) -> None:
    assert _direct_source_audit_findings(source, shared=True)


@pytest.mark.parametrize(
    "source",
    (
        """
def legacy_projection(value):
    import hashlib as local_hashlib

    return local_hashlib.new("sha256", value)

def project_direct_payload(value):
    return {"payload": value}
""",
        """
def legacy_projection(value):
    from provenance import canonical_sha256 as summarize

    return summarize(value)

def project_direct_payload(value):
    return {"payload": value}
""",
    ),
)
def test_direct_source_audit_ignores_function_local_imports_in_legacy_siblings(
    source: str,
) -> None:
    assert _direct_source_audit_findings(source, shared=True) == ()


def test_scoped_direct_source_and_schema_migration_audit() -> None:
    tuning = json.loads((ROOT / "study/comparator_tuning.json").read_text())
    contract = json.loads((ROOT / "study/selection_contract.json").read_text())
    search = json.loads((ROOT / "study/development_search.json").read_text())
    tracked_sections = (
        tuning,
        {
            "comparator_tuning": contract["comparator_tuning"],
            "comparator_method_bindings": contract["comparator_method_bindings"],
        },
        {"comparator_tuning": search["authority"]["comparator_tuning"]},
    )
    assert not any(
        _forbidden_direct_key(key)
        for section in tracked_sections
        for key in _direct_schema_keys(section)
    )

    direct_modules = (
        "maskimpute_benchmark/comparator_tuning.py",
        "maskimpute_benchmark/fair_comparator_plan.py",
        "maskimpute_benchmark/fair_comparator_execution.py",
        "maskimpute_benchmark/fair_comparator_checkpoint.py",
    )
    for relative in direct_modules:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert _direct_source_audit_findings(source, shared=False) == (), relative

    shared_modules = (
        "maskimpute_benchmark/runner.py",
        "maskimpute_benchmark/development_evaluation.py",
        "maskimpute_benchmark/downstream_evidence.py",
        "maskimpute_benchmark/selection.py",
    )
    for relative in shared_modules:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert _direct_source_audit_findings(source, shared=True) == (), relative


def test_scoped_direct_schema_audit_covers_every_synthetic_artifact() -> None:
    artifacts = _synthetic_direct_artifacts()

    assert tuple(artifacts) == (
        "request",
        "plan",
        "attempt",
        "checkpoint",
        "projection",
    )
    assert not any(
        _forbidden_direct_key(key)
        for artifact in artifacts.values()
        for key in _direct_schema_keys(artifact)
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _authority_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for relative in AUTHORITY_FILES:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return repository


def _refresh_legacy_authority_references(repository: Path) -> None:
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    authority = ledger["authority"]
    authority["protocol_sha256"] = hashlib.sha256(
        (repository / "study/protocol.json").read_bytes()
    ).hexdigest()
    authority["development_panel_sha256"] = hashlib.sha256(
        (repository / "study/development_panel.json").read_bytes()
    ).hexdigest()
    authority["methods_sha256"] = hashlib.sha256(
        (repository / "study/methods.json").read_bytes()
    ).hexdigest()
    authority["selection_contract_sha256"] = hashlib.sha256(
        (repository / "study/selection_contract.json").read_bytes()
    ).hexdigest()
    authority["ablations_sha256"] = hashlib.sha256(
        (repository / "study/ablations.json").read_bytes()
    ).hexdigest()
    authority["calibration_contract_sha256"] = hashlib.sha256(
        (repository / "study/calibration_contract.json").read_bytes()
    ).hexdigest()
    _write_json(ledger_path, ledger)


METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
    "null_de_fpr",
)


def test_selection_contract_binds_comparator_authority_by_direct_reference() -> None:
    contract = json.loads((ROOT / "study/selection_contract.json").read_text())
    reference = contract["comparator_tuning"]
    assert reference == {
        "path": "study/comparator_tuning.json",
        "schema_version": 2,
        "authority_revision": "fair-comparator-direct-v1",
    }


def test_development_search_binds_comparator_authority_by_direct_reference() -> None:
    ledger = json.loads((ROOT / "study/development_search.json").read_text())
    reference = ledger["authority"]["comparator_tuning"]
    assert reference == {
        "path": "study/comparator_tuning.json",
        "schema_version": 2,
        "authority_revision": "fair-comparator-direct-v1",
    }


def test_selection_authority_carries_direct_comparator_reference() -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.selection import _load_selection_authority

    authority = _load_selection_authority(ROOT, require_clean=False)
    assert authority.comparator_tuning == ComparatorAuthorityReference(
        path="study/comparator_tuning.json",
        schema_version=2,
        authority_revision="fair-comparator-direct-v1",
    )
    assert "study/comparator_tuning.json" not in authority.file_sha256
    assert not hasattr(authority, "comparator_tuning_file_sha256")
    assert not hasattr(authority, "comparator_tuning_payload_sha256")
    assert tuple(authority.comparator_method_bindings) == (
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "scziva",
        "afmf",
        "biaeimpute",
        "sccr",
        "scsdae",
    )
    with pytest.raises(TypeError):
        authority.comparator_method_bindings["magic"] = (
            authority.comparator_method_bindings["magic"]
        )


def test_direct_selected_projection_rejects_duplicate_or_drifted_authority_rows() -> (
    None
):
    from maskimpute_benchmark.comparator_tuning import (
        ComparatorAuthorityReference,
        load_comparator_tuning_authority,
    )
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        project_direct_selected_comparators,
    )

    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    reference = ComparatorAuthorityReference(
        path="study/comparator_tuning.json",
        schema_version=authority.schema_version,
        authority_revision=authority.authority_revision,
    )
    row = authority.configurations_for("magic")[0]
    with pytest.raises(SelectionAuthorityError, match="methods must be unique"):
        project_direct_selected_comparators(
            reference,
            authority,
            (row, row),
        )
    drifted = replace(row, payload_json=row.payload_json.replace('"knn":5', '"knn":7'))
    with pytest.raises(SelectionAuthorityError, match="exact authority evidence"):
        project_direct_selected_comparators(
            reference,
            authority,
            (drifted,),
        )
    with pytest.raises(SelectionAuthorityError, match="authority reference differs"):
        project_direct_selected_comparators(
            replace(reference, schema_version=3),
            authority,
            (row,),
        )
    with pytest.raises(SelectionAuthorityError, match="authority reference differs"):
        project_direct_selected_comparators(
            replace(reference, schema_version=2.0),
            authority,
            (row,),
        )


@pytest.mark.parametrize(
    "mutation",
    ("schema-version", "authority-revision", "closed-authority-field"),
)
def test_direct_selected_projection_rejects_coherently_forged_authority(
    mutation: str,
) -> None:
    from maskimpute_benchmark.comparator_tuning import (
        ComparatorAuthorityReference,
        load_comparator_tuning_authority,
    )
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        project_direct_selected_comparators,
    )

    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    if mutation == "schema-version":
        forged = replace(authority, schema_version=3)
    elif mutation == "authority-revision":
        forged = replace(authority, authority_revision="fair-comparator-direct-v2")
    else:
        forged = replace(authority, contract_id="forged-comparator-contract")
    reference = ComparatorAuthorityReference(
        path="study/comparator_tuning.json",
        schema_version=forged.schema_version,
        authority_revision=forged.authority_revision,
    )

    with pytest.raises(SelectionAuthorityError, match="authority"):
        project_direct_selected_comparators(
            reference,
            forged,
            forged.configurations_for("magic")[:1],
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "authority-revision",
        "authority-path",
        "schema-version",
        "full-payload",
        "row-order",
    ),
)
def test_selection_authority_rejects_coherently_reencoded_direct_linkage_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        _load_selection_authority,
    )

    repository = _authority_repository(tmp_path)
    contract_path = repository / "study/selection_contract.json"
    ledger_path = repository / "study/development_search.json"
    tuning_path = repository / "study/comparator_tuning.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))

    if mutation == "authority-revision":
        tuning["authority_revision"] = "fair-comparator-direct-v2"
        contract["comparator_tuning"]["authority_revision"] = (
            "fair-comparator-direct-v2"
        )
        ledger["authority"]["comparator_tuning"]["authority_revision"] = (
            "fair-comparator-direct-v2"
        )
    elif mutation == "authority-path":
        contract["comparator_tuning"]["path"] = "study/other-comparator.json"
        ledger["authority"]["comparator_tuning"]["path"] = "study/other-comparator.json"
    elif mutation == "schema-version":
        tuning["schema_version"] = 3
        contract["comparator_tuning"]["schema_version"] = 3
        ledger["authority"]["comparator_tuning"]["schema_version"] = 3
    elif mutation == "full-payload":
        tuning["configurations"][1]["payload"]["diffusion_time"] = 2
    elif mutation == "row-order":
        rows = tuning["configurations"]
        rows[1], rows[2] = rows[2], rows[1]
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    _write_json(tuning_path, tuning)
    _write_json(contract_path, contract)
    _write_json(ledger_path, ledger)
    _refresh_legacy_authority_references(repository)

    with pytest.raises(SelectionAuthorityError):
        _load_selection_authority(repository, require_clean=False)


def test_selection_authority_rejects_coherent_comparator_resource_projection_drift(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        _load_selection_authority,
    )

    repository = _authority_repository(tmp_path)
    methods_path = repository / "study/methods.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    magic = next(row for row in methods["methods"] if row["id"] == "magic")
    magic["resources"]["timeout_seconds"] += 1
    _write_json(methods_path, methods)
    _refresh_legacy_authority_references(repository)

    with pytest.raises(SelectionAuthorityError, match="method binding"):
        _load_selection_authority(repository, require_clean=False)


def test_direct_reference_migration_preserves_unrelated_legacy_authority_values() -> (
    None
):
    base = "3258fad89226954295fd7b9cfd097401e33637f5"

    def at_base(relative: str) -> dict[str, object]:
        raw = subprocess.run(
            ["git", "show", f"{base}:{relative}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return json.loads(raw)

    old_contract = at_base("study/selection_contract.json")
    new_contract = json.loads(
        (ROOT / "study/selection_contract.json").read_text(encoding="utf-8")
    )
    old_contract.pop("comparator_tuning_path")
    old_contract.pop("comparator_tuning_file_sha256")
    old_contract.pop("comparator_tuning_payload_sha256")
    new_contract.pop("comparator_tuning")
    new_contract.pop("comparator_method_bindings")
    assert json.dumps(new_contract, separators=(",", ":")) == json.dumps(
        old_contract, separators=(",", ":")
    )

    old_ledger = at_base("study/development_search.json")
    new_ledger = json.loads(
        (ROOT / "study/development_search.json").read_text(encoding="utf-8")
    )
    old_ledger["authority"].pop("comparator_tuning_file_sha256")
    old_ledger["authority"].pop("comparator_tuning_payload_sha256")
    new_ledger["authority"].pop("comparator_tuning")
    new_ledger["authority"]["selection_contract_sha256"] = old_ledger["authority"][
        "selection_contract_sha256"
    ]
    assert json.dumps(new_ledger, separators=(",", ":")) == json.dumps(
        old_ledger, separators=(",", ":")
    )
    tracked_selection_bytes = (ROOT / "study/selection_contract.json").read_bytes()
    assert (
        json.loads((ROOT / "study/development_search.json").read_text())["authority"][
            "selection_contract_sha256"
        ]
        == hashlib.sha256(tracked_selection_bytes).hexdigest()
    )


def _dataset_rows():
    rows = []
    for mechanism in ("symsim", "sergio", "sparsim", "semisynthetic"):
        for draw in ("draw-01", "draw-02"):
            for view in ("moderate", "severe"):
                label = f"{mechanism}:{draw}:{view}"
                dataset_id = (
                    f"dataset-{hashlib.sha256(label.encode()).hexdigest()[:24]}"
                )
                rows.append(
                    {
                        "mechanism": mechanism,
                        "biological_id": draw,
                        "technical_view": view,
                        "dataset_id": dataset_id,
                        "dataset_sha256": hashlib.sha256(
                            f"dataset:{label}".encode()
                        ).hexdigest(),
                        "output_file_sha256": hashlib.sha256(
                            f"output:{label}".encode()
                        ).hexdigest(),
                        "truth_sha256": hashlib.sha256(
                            f"truth:{mechanism}:{draw}".encode()
                        ).hexdigest(),
                        "output_path": f"dev/datasets/{mechanism}/{draw}/{view}.h5ad",
                        "independent_unit_id": f"{mechanism}:{draw}",
                        "cells": 900,
                        "genes": 500,
                        "status": "completed",
                    }
                )
    return rows


def _ready_repository(tmp_path: Path):
    from maskimpute.calibration import CalibrationRecord, fit_development_calibration
    from maskimpute_benchmark.selection import _canonical_sha256

    repository = tmp_path / "repository"
    for relative in (
        "study/protocol.json",
        "study/development_panel.json",
        "study/methods.json",
        "study/ablations.json",
        "study/calibration_contract.json",
        "study/comparator_tuning.json",
        "study/selection_contract.json",
        "study/development_search.json",
    ):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, destination)
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    contract_path = repository / "study/selection_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )
    ledger["authority"]["selection_contract_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    entries = []
    for row in _dataset_rows():
        label = ":".join(
            (row["mechanism"], row["biological_id"], row["technical_view"])
        )
        entries.append(
            {
                "mechanism": row["mechanism"],
                "biological_id": row["biological_id"],
                "technical_view": row["technical_view"],
                "dataset_id": row["dataset_id"],
                "dataset_sha256": row["dataset_sha256"],
                "input_sha256": hashlib.sha256(f"input:{label}".encode()).hexdigest(),
                "cell_ids_sha256": hashlib.sha256(
                    f"cells:{row['mechanism']}:{row['biological_id']}".encode()
                ).hexdigest(),
                "excluded_cell_count": 0,
                "excluded_cell_ids_sha256": hashlib.sha256(
                    b"empty-cell-set"
                ).hexdigest(),
                "retained_cell_count": 900,
                "retained_cell_ids_sha256": hashlib.sha256(
                    f"cells:{row['mechanism']}:{row['biological_id']}".encode()
                ).hexdigest(),
                "score_sha256": hashlib.sha256(f"score:{label}".encode()).hexdigest(),
                "config_sha256": contract["count_model_config_sha256"],
            }
        )
    score_manifest = (
        repository / "artifacts/study/development/count_scores/manifest.json"
    )
    score_manifest.parent.mkdir(parents=True, exist_ok=True)
    score_core = {
        "schema_version": 1,
        "artifact_type": "maskimpute_development_count_score_manifest",
        "dataset_manifest_sha256": "a" * 64,
        "count_model_config_sha256": contract["count_model_config_sha256"],
        "dataset_qc_policy_sha256": contract["dataset_qc_policy_sha256"],
        "entries": entries,
    }
    score_payload = {
        **score_core,
        "manifest_sha256": _canonical_sha256(score_core),
    }
    score_manifest.write_text(
        json.dumps(score_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ledger["count_score_manifest"] = {
        "status": "ready",
        "path": "artifacts/study/development/count_scores/manifest.json",
        "sha256": hashlib.sha256(score_manifest.read_bytes()).hexdigest(),
    }

    score_by_unit = {
        (row["mechanism"], row["biological_id"], row["technical_view"]): row
        for row in entries
    }
    calibration_records = []
    for index, row in enumerate(
        (item for item in _dataset_rows() if item["mechanism"] == "symsim"),
        start=1,
    ):
        score = score_by_unit[
            (row["mechanism"], row["biological_id"], row["technical_view"])
        ]
        calibration_records.append(
            CalibrationRecord(
                p_pre_zero=(0.1, 0.25, 0.7, 0.9),
                target=(0, 0, 1, 1),
                mechanism="symsim",
                biological_id=row["biological_id"],
                manifest_sha256=score["score_sha256"],
                truth_kind="exact_pre_capture",
                namespace="dev",
                data_role="development",
                technical_view=row["technical_view"],
                dataset_id=row["dataset_id"],
                dataset_sha256=row["dataset_sha256"],
                protocol_sha256=ledger["authority"]["protocol_sha256"],
            )
        )
    artifact = fit_development_calibration(calibration_records)
    calibration = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    calibration.parent.mkdir(parents=True, exist_ok=True)
    calibration.write_text(
        json.dumps(artifact.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    calibration_sha = hashlib.sha256(calibration.read_bytes()).hexdigest()
    ledger["retained_calibration_artifact"] = {
        "status": "ready",
        "path": ("artifacts/study/development/calibration/retained_calibration.json"),
        "sha256": calibration_sha,
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return repository, calibration_sha


def _source_evidence(repository: Path):
    from dataclasses import asdict

    from maskimpute_benchmark.development_evaluation import (
        validate_real_source_artifacts,
    )
    from maskimpute_benchmark.sources import load_source_ledger

    source_specs = (
        ("baron-pancreas-umi", "semisynthetic_source", "baron.dat"),
        ("cite-seq-cbmc-rna-protein", "orthogonal_validation", "cbmc.dat"),
        ("tung-ipsc-ercc-bulk-replicates", "orthogonal_validation", "tung.dat"),
    )
    sources = []
    for index, (source_id, role, name) in enumerate(source_specs, start=1):
        raw = f"source-{index}\n".encode()
        digest = hashlib.sha256(raw).hexdigest()
        artifact = repository / "artifacts/external/data" / source_id / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(raw)
        sources.append(
            {
                "id": source_id,
                "role": role,
                "mechanism": "semisynthetic" if index == 1 else None,
                "source_type": "data",
                "url": f"https://example.org/{source_id}",
                "revision": f"GSE{index}:2026-07-12",
                "license": "CC0-1.0",
                "license_url": "https://example.org/license",
                "citation_doi": f"10.1234/source.{index}",
                "expected_checksum": None,
                "eligibility": "eligible",
                "endpoints": ["source_validation"],
                "artifacts": [
                    {
                        "name": name,
                        "url": f"https://example.org/{name}",
                        "expected_checksum": {
                            "algorithm": "sha256",
                            "value": digest,
                        },
                    }
                ],
            }
        )
    ledger_path = repository / "study/sources.json"
    ledger_path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
    ledger = load_source_ledger(ledger_path)
    for source in sources:
        artifact = source["artifacts"][0]
        artifact_path = (
            repository / "artifacts/external/data" / source["id"] / artifact["name"]
        )
        receipt = {
            "schema_version": 1,
            "source_id": source["id"],
            "role": source["role"],
            "source_type": "data",
            "source_url": source["url"],
            "revision": source["revision"],
            "resolved_revision": source["revision"],
            "license": source["license"],
            "citation_doi": source["citation_doi"],
            "verified_checksum": None,
            "ledger_sha256": ledger.sha256,
            "artifacts": [
                {
                    "name": artifact["name"],
                    "sha256": artifact["expected_checksum"]["value"],
                    "size_bytes": artifact_path.stat().st_size,
                }
            ],
        }
        receipt_path = (
            repository / "artifacts/external/receipts" / f"{source['id']}.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
    evidence = validate_real_source_artifacts(repository)
    return {
        "ledger_path": evidence.ledger_path,
        "ledger_file_sha256": evidence.ledger_file_sha256,
        "ledger_sha256": evidence.ledger_sha256,
        "receipts": [asdict(value) for value in evidence.receipts],
        "artifacts": [asdict(value) for value in evidence.artifacts],
    }


def _status_and_payload(authority):
    from maskimpute_benchmark.selection import _canonical_sha256

    datasets = _dataset_rows()
    dataset_by_unit = {
        (row["mechanism"], row["biological_id"], row["technical_view"]): row
        for row in datasets
    }
    selected_methods = [
        declaration
        for declaration in authority.declarations
        if declaration.id in authority.scheduled_same_input_ids
        or declaration.role == "candidate"
    ]
    records = []
    for declaration in selected_methods:
        if declaration.role == "observed_control":
            base = 1.4
        elif declaration.role == "candidate":
            base = 0.8
        else:
            base = 1.0
        seeds = authority.model_seeds if declaration.stochastic else (None,)
        for metric in METRICS:
            mechanisms = (
                ("symsim",)
                if metric == "mse_pre_dropout_zero"
                else authority.mechanisms
            )
            for mechanism in mechanisms:
                for draw in authority.biological_ids:
                    for view in authority.technical_views:
                        dataset = dataset_by_unit[(mechanism, draw, view)]
                        for seed in seeds:
                            value = 0.05 if metric == "null_de_fpr" else base
                            records.append(
                                {
                                    "mechanism": mechanism,
                                    "biological_id": draw,
                                    "technical_view": view,
                                    "dataset_id": dataset["dataset_id"],
                                    "dataset_sha256": dataset["dataset_sha256"],
                                    "method": declaration.id,
                                    "method_sha256": authority.method_bindings[
                                        declaration.id
                                    ],
                                    "model_seed": seed,
                                    "metric": metric,
                                    "value": value,
                                    "status": "completed",
                                }
                            )
    intervals = [
        {
            "configuration": attempt.configuration_id,
            "endpoint": endpoint.id,
            "comparison": "observed",
            "estimate": 0.0,
            "ci_lower": -0.01,
            "ci_upper": 0.01,
            "status": "completed",
        }
        for attempt in authority.attempts
        for endpoint in authority.endpoint_policies
    ]
    manifest_sha = "a" * 64
    status = {
        "schema_version": 1,
        "namespace": "dev",
        "status": "completed",
        "completed_count": 16,
        "failed_count": 0,
        "independent_unit_count": 8,
        "manifest_sha256": manifest_sha,
        "protocol_sha256": authority.file_sha256["study/protocol.json"],
        "design_sha256": "f" * 64,
        "seed_source_sha256": "9" * 64,
        "execution_claim_id": None,
        "round_id": None,
        "rows": datasets,
    }
    core = {
        "schema_version": 2,
        "dataset_manifest_sha256": manifest_sha,
        "count_score_manifest_sha256": authority.count_score_manifest.sha256,
        "retained_calibration_artifact_sha256": (authority.retained_calibration.sha256),
        "evaluation_manifest_sha256": "e" * 64,
        "records": records,
        "orthogonal_intervals": intervals,
    }
    payload = {**core, "result_sha256": _canonical_sha256(core)}
    return status, payload


def _attach_evaluation_manifest(
    repository,
    payload,
    *,
    corrupt=None,
    sources=None,
    reconstruction=None,
    orthogonal=None,
    null_de_audits=None,
    orthogonal_audits=None,
):
    from maskimpute_benchmark.selection import _canonical_sha256

    evidence_core = {
        key: value
        for key, value in payload.items()
        if key not in {"result_sha256", "evaluation_manifest_sha256"}
    }
    evaluation_core = {
        "schema_version": 1,
        "artifact_type": "maskimpute_development_selection_evaluation_manifest",
        "selection_evidence_sha256": _canonical_sha256(evidence_core),
        "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "count_score_manifest": {
            "path": "artifacts/study/development/count_scores/manifest.json",
            "file_sha256": payload["count_score_manifest_sha256"],
        },
        "retained_calibration_artifact": {
            "path": (
                "artifacts/study/development/calibration/retained_calibration.json"
            ),
            "file_sha256": payload["retained_calibration_artifact_sha256"],
        },
        "reconstruction": {} if reconstruction is None else reconstruction,
        "orthogonal": {} if orthogonal is None else orthogonal,
        "sources": {} if sources is None else sources,
        "null_de_audits": [] if null_de_audits is None else null_de_audits,
        "orthogonal_audits": ([] if orthogonal_audits is None else orthogonal_audits),
        "combined_score": None,
    }
    evaluation = {
        **evaluation_core,
        "manifest_sha256": _canonical_sha256(evaluation_core),
    }
    if corrupt is not None:
        corrupt(evaluation)
    path = (
        repository / "artifacts/study/development/evaluation/evaluation_manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evaluation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    result_core = {
        key: value for key, value in payload.items() if key != "result_sha256"
    }
    result_core["evaluation_manifest_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    return {**result_core, "result_sha256": _canonical_sha256(result_core)}


def _minimal_reconstruction_evidence(repository: Path, *, input_hashes=None):
    from maskimpute_benchmark.selection import _canonical_sha256

    checkpoint_core = {
        "schema_version": 1,
        "plan_sha256": "a" * 64,
        "input_hashes": (
            {"dataset_manifest_sha256": "0" * 64}
            if input_hashes is None
            else input_hashes
        ),
        "planned_run_count": 0,
        "status": "completed",
        "evaluation_scope": "reconstruction_only",
        "comparator_selection_status": "complete_terminal_denominator",
        "selection_complete": False,
        "selection_blockers": [],
        "records": [],
        "budget": {},
    }
    checkpoint = {
        **checkpoint_core,
        "checkpoint_sha256": _canonical_sha256(checkpoint_core),
    }
    path = (
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return {
        "checkpoint_path": (
            "artifacts/study/development/competition-reconstruction/checkpoint.json"
        ),
        "checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "plan_sha256": checkpoint["plan_sha256"],
        "input_hashes": checkpoint["input_hashes"],
        "raw_artifacts": [],
    }


def _reconstruction_inputs(authority, status, payload):
    return {
        "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "dataset_design_sha256": status["design_sha256"],
        "dataset_seed_source_sha256": status["seed_source_sha256"],
        "protocol_sha256": authority.file_sha256["study/protocol.json"],
        "method_registry_sha256": authority.file_sha256["study/methods.json"],
        "selection_contract_sha256": authority.file_sha256[
            "study/selection_contract.json"
        ],
        "development_search_sha256": authority.file_sha256[
            "study/development_search.json"
        ],
        "ablation_registry_sha256": authority.file_sha256["study/ablations.json"],
        "runner_authority_sha256": "8" * 64,
        "execution_environment_sha256": "7" * 64,
        "base_configuration_sha256": authority.base_maskimpute_config_sha256,
        "count_model_config_sha256": authority.count_model_config_sha256,
        "dataset_qc_policy_sha256": authority.dataset_qc_policy_sha256,
        "count_score_manifest_sha256": payload["count_score_manifest_sha256"],
        "retained_calibration_sha256": payload["retained_calibration_artifact_sha256"],
    }


def _minimal_orthogonal_evidence(repository: Path):
    from maskimpute_benchmark.selection import _canonical_sha256

    output_relative = "outputs/source-test--observed--deterministic.log2-cp10k-f64.zlib"
    root = repository / "artifacts/study/development/evaluation/orthogonal"
    output_path = root / output_relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output = (1).to_bytes(8, "little")
    compressed_output = zlib.compress(raw_output, level=6)
    output_path.write_bytes(compressed_output)
    record = {
        "source_id": "source-test",
        "configuration": "observed",
        "configuration_sha256": "0" * 64,
        "model_seed": None,
        "method_input_sha256": "1" * 64,
        "status": "completed",
        "reason": None,
        "output_path": output_relative,
        "output_file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "output_compressed_nbytes": len(compressed_output),
        "output_encoding": "zlib_raw_f64_v1",
        "output_uncompressed_nbytes": len(raw_output),
        "output_uncompressed_sha256": hashlib.sha256(raw_output).hexdigest(),
        "output_shape": [1, 1],
        "output_dtype": "<f8",
        "output_scale": "log2_cp10k_plus_1",
    }
    manifest_core = {
        "schema_version": 2,
        "artifact_type": "maskimpute_orthogonal_method_outputs",
        "authority": {
            "inputs": [
                {
                    "source_id": "source-test",
                    "source_dataset_sha256": "9" * 64,
                    "method_input_sha256": "1" * 64,
                    "shape": [1, 1],
                    "cell_ids_sha256": "2" * 64,
                    "gene_ids_sha256": "3" * 64,
                }
            ],
            "configurations": [],
            "model_seeds": [],
            "artifact_bindings": {},
        },
        "status": "completed",
        "planned_record_count": 1,
        "records": [record],
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": _canonical_sha256(manifest_core),
    }
    manifest_path = root / "orthogonal_outputs.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return {
        "manifest_path": (
            "artifacts/study/development/evaluation/orthogonal/orthogonal_outputs.json"
        ),
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest["manifest_sha256"],
        "records": [record],
    }, output_path


def test_public_selection_api_accepts_results_only_not_design_authority():
    from maskimpute_benchmark.selection import select_development_candidate

    signature = inspect.signature(select_development_candidate)

    assert tuple(signature.parameters) == ("payload",)


def test_tracked_ledger_atomically_invalidates_old_score_and_calibration():
    calibration_path = Path("study/calibration_contract.json")
    selection_path = Path("study/selection_contract.json")
    selection = json.loads(selection_path.read_text())
    ledger = json.loads(Path("study/development_search.json").read_text())

    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert selection["calibration_contract_sha256"] == calibration_sha
    assert ledger["authority"]["calibration_contract_sha256"] == calibration_sha
    assert ledger["authority"]["selection_contract_sha256"] == selection_sha

    assert ledger["count_score_manifest"] == {
        "status": "pending",
        "path": "artifacts/study/development/count_scores/manifest.json",
        "sha256": None,
    }
    assert ledger["retained_calibration_artifact"] == {
        "status": "pending",
        "path": "artifacts/study/development/calibration/retained_calibration.json",
        "sha256": None,
    }


def test_repository_authority_derives_design_methods_and_pending_artifacts():
    from maskimpute_benchmark.selection import (
        _load_selection_authority,
        load_publication_execution_authority,
    )

    authority = _load_selection_authority(Path.cwd(), require_clean=False)

    assert authority.mechanisms == (
        "symsim",
        "sergio",
        "sparsim",
        "semisynthetic",
    )
    assert authority.biological_ids == ("draw-01", "draw-02")
    assert authority.technical_views == ("moderate", "severe")
    assert authority.model_seeds == (42, 43, 44)
    assert authority.required_comparator_ids == (
        "observed",
        "capacity-matched-ae",
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "scziva",
        "afmf",
        "sccr",
        "scsdae",
    )
    assert authority.retained_calibration.status == "pending"
    assert authority.retained_calibration.path == (
        "artifacts/study/development/calibration/retained_calibration.json"
    )
    assert authority.retained_calibration.sha256 is None
    assert authority.count_score_manifest.status == "pending"
    assert authority.count_score_manifest.path == (
        "artifacts/study/development/count_scores/manifest.json"
    )
    assert authority.count_score_manifest.sha256 is None
    assert dict(authority.base_maskimpute_config) == {
        "hidden_dims": (128, 64),
        "latent_dim": 24,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 300,
        "patience": 30,
        "artificial_mask_fraction": 0.2,
        "validation_fraction": 0.1,
        "log_count_bin_edges": (
            1.0986122886681096,
            2.1972245773362196,
            3.4965075614664802,
        ),
        "early_stopping_min_delta": 0.0,
        "pre_zero_regularization": 1.0,
        "gate_gamma": 1.0,
        "normalization_target": 10000.0,
    }
    assert dict(authority.count_model_config) == {
        "n_folds": 5,
        "use_library_size_exposure": True,
        "mean_prior_strength": 1.0,
        "mean_floor": 1e-8,
        "dispersion_prior_strength": 10.0,
        "link_bins": 64,
        "link_max_iter": 200,
        "link_tolerance": 1e-10,
        "link_bound": 30.0,
    }
    assert authority.ablation_spec_ids == (
        "maskimpute-reference",
        "capacity-matched-ae",
        "no-gate",
        "no-pre-zero-regularizer",
        "no-explicit-mask",
        "full-denoising",
        "direct-score",
    )
    assert len(authority.ablation_run_keys) == 21
    assert authority.ablation_run_keys[:3] == (
        ("maskimpute-reference", 42),
        ("maskimpute-reference", 43),
        ("maskimpute-reference", 44),
    )
    assert authority.calibration_equivalence_reason is None
    assert authority.calibration_effect_status == "pending_retained_artifact"
    assert dict(authority.dataset_qc_policy) == {
        "cell_exclusion_rule": "observed_library_size_equals_zero",
        "minimum_retained_cells": 2,
        "application": (
            "pre_dispatch_pair_union_zero_library_identical_cell_subset_all_methods"
        ),
        "additional_cell_filtering": "forbidden",
        "gene_filtering": "forbidden",
        "required_audit_fields": (
            "excluded_cell_count",
            "excluded_cell_ids_sha256",
            "retained_cell_count",
            "retained_cell_ids_sha256",
        ),
    }
    assert authority.dataset_qc_policy_sha256 == (
        "81dc2ecd1749d9390e499ae21fabb8d3b08f40eec58334c860cd9a23dd4fc2d7"
    )
    assert (
        tuple(inspect.signature(load_publication_execution_authority).parameters) == ()
    )
    assert authority.file_sha256["study/methods.json"] == (
        "d71197d5d94d5009c807c57b393372fefa56f0f48327558bae334ab68875588d"
    )
    assert authority.file_sha256["study/ablations.json"] == (
        "dd4da34e0ebe5e7eb349fac3ed89063781bcddf640b01601b9a3c82a2e43b26f"
    )
    assert authority.file_sha256["study/calibration_contract.json"] == (
        "180d85cc18e359970fff3c9cff37190c2b944b13b0883a46be2765c439a8a1b3"
    )
    assert len(authority.attempts) == 20
    assert tuple(item.configuration_id for item in authority.attempts)[:2] == (
        "v27-c01-direct-r1-g1",
        "v27-c02-calibrated-r1-g0p5",
    )
    assert tuple(item.configuration_id for item in authority.attempts)[-2:] == (
        "v27-c19-calibrated-r10-g2",
        "v27-c20-calibrated-r10-g3",
    )
    assert tuple(item.configuration_id for item in authority.exclusions) == (
        "v27-c21-calibrated-r10-g4",
        "v27-c22-calibrated-r10-g6",
    )
    assert all(
        item.reason_code == "exploratory_budget_overrun_not_selection_eligible"
        for item in authority.exclusions
    )


def test_selection_authority_uses_exact_comparator_readiness_sets() -> None:
    from maskimpute_benchmark.selection import _load_selection_authority

    authority = _load_selection_authority(Path.cwd(), require_clean=False)
    assert authority.scheduled_same_input_ids == (
        "observed",
        "capacity-matched-ae",
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "scziva",
        "afmf",
        "biaeimpute",
        "sccr",
        "scsdae",
    )
    assert authority.required_control_ids == ("observed", "capacity-matched-ae")
    assert authority.established_comparator_ids == (
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
    )
    assert authority.modern_core_ids == ("scziva", "afmf", "biaeimpute", "sccr")
    assert "biaeimpute" in authority.scheduled_same_input_ids
    assert authority.comparator_tuning.path == "study/comparator_tuning.json"
    assert authority.comparator_tuning.schema_version == 2
    assert authority.comparator_tuning.authority_revision == "fair-comparator-direct-v1"
    assert all(
        not declaration.required_for_claim
        for declaration in authority.declarations
        if declaration.role != "candidate"
    )


def test_selection_authority_rejects_biaeimpute_omission(tmp_path: Path) -> None:
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        _load_selection_authority,
    )

    repository, _calibration_sha = _ready_repository(tmp_path)
    contract_path = repository / "study/selection_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["scheduled_same_input_ids"].remove("biaeimpute")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    with pytest.raises(
        SelectionAuthorityError, match="scheduled same-input denominator"
    ):
        _load_selection_authority(repository, require_clean=False)


def test_public_selection_rejects_malformed_result_before_evidence_validation(tmp_path):
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        _select_for_repository,
    )

    repository, _calibration_sha = _ready_repository(tmp_path)

    with pytest.raises(SelectionAuthorityError, match="missing or extra fields"):
        _select_for_repository({}, repository, require_clean=False)


def test_ready_public_selection_binds_results_to_all_repository_authorities(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection

    repository, calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    monkeypatch.setattr(
        evaluation,
        "validate_selection_evaluation_manifest",
        lambda *_args: SimpleNamespace(bindings={}),
    )
    report = selection._select_for_repository(
        payload,
        repository,
        require_clean=False,
    )

    assert report.selected_configuration == "v27-c01-direct-r1-g1"
    assert tuple(item.configuration_id for item in report.excluded_configurations) == (
        "v27-c21-calibrated-r10-g4",
        "v27-c22-calibrated-r10-g6",
    )
    assert report.authority_bindings is not None
    assert report.authority_bindings["retained_calibration_artifact_sha256"] == (
        calibration_sha
    )
    assert (
        report.authority_bindings["development_result_sha256"]
        == payload["result_sha256"]
    )
    assert (
        report.authority_bindings["dataset_manifest_sha256"]
        == (status["manifest_sha256"])
    )
    calibration_payload = json.loads(
        (
            repository
            / "artifacts/study/development/calibration/retained_calibration.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        report.authority_bindings["retained_calibration_algorithm"]
        == (calibration_payload["selected_algorithm"])
    )
    if calibration_payload["selected_algorithm"] == "identity":
        assert report.authority_bindings["calibration_equivalence_reason"] == (
            "retained_identity_calibrator_equals_direct_score"
        )
    else:
        assert report.authority_bindings["calibration_equivalence_reason"] == (
            "retained_nonidentity_calibrator_transformed_score"
        )


def test_schema_four_selection_requires_downstream_and_revalidates_legacy_envelope(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    stage_paths = development_selection_stage_paths(None)
    source_path = repository / stage_paths.source_selection_input
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    source_path.write_bytes(source_raw)
    source_file_sha = hashlib.sha256(source_raw).hexdigest()
    core = {key: value for key, value in payload.items() if key != "result_sha256"}
    core.update(
        {
            "schema_version": 4,
            "revision_versions": [],
            "downstream_evidence": {
                "path": stage_paths.downstream_directory,
                "source_selection_input_path": (stage_paths.source_selection_input),
                "source_selection_input_file_sha256": source_file_sha,
                "source_selection_result_sha256": payload["result_sha256"],
                "manifest_file_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "plan_sha256": "3" * 64,
                "planned_denominator_count": 1,
                "endpoint_row_count": 8,
            },
        }
    )
    schema_four = {**core, "result_sha256": selection._canonical_sha256(core)}
    observed_evaluation_data = []
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    monkeypatch.setattr(
        selection,
        "validate_downstream_selection_completeness",
        lambda *_args: {
            "downstream_manifest_sha256": "2" * 64,
            "downstream_plan_sha256": "3" * 64,
        },
    )

    def validate_legacy(_repository, data, _authority, _status):
        observed_evaluation_data.append(data)
        return SimpleNamespace(bindings={})

    monkeypatch.setattr(
        evaluation,
        "validate_selection_evaluation_manifest",
        validate_legacy,
    )

    report = selection._select_for_repository(
        schema_four,
        repository,
        require_clean=False,
    )

    assert report.selected_configuration == "v27-c01-direct-r1-g1"
    assert report.authority_bindings["downstream_plan_sha256"] == "3" * 64
    assert len(observed_evaluation_data) == 1
    projected = observed_evaluation_data[0]
    assert projected["schema_version"] == 2
    assert "downstream_evidence" not in projected
    assert "revision_versions" not in projected
    projected_core = {
        key: value for key, value in projected.items() if key != "result_sha256"
    }
    assert projected["result_sha256"] == selection._canonical_sha256(projected_core)

    forged_binding = dict(schema_four["downstream_evidence"])
    forged_binding["source_selection_input_file_sha256"] = "0" * 64
    forged_core = {
        **{
            key: value
            for key, value in schema_four.items()
            if key not in {"downstream_evidence", "result_sha256"}
        },
        "downstream_evidence": forged_binding,
    }
    forged = {
        **forged_core,
        "result_sha256": selection._canonical_sha256(forged_core),
    }
    with pytest.raises(
        selection.SelectionAuthorityError,
        match="promoted selection source differs",
    ):
        selection._select_for_repository(
            forged,
            repository,
            require_clean=False,
        )


def test_revision_downstream_sources_crosscheck_each_evaluation_checkpoint() -> None:
    import maskimpute_benchmark.selection as selection

    downstream: dict[str, str] = {}
    evaluation: dict[str, str] = {}
    for source_id in ("base", "v28", "v29"):
        for downstream_name, evaluation_name in (
            ("checkpoint_path", "reconstruction_checkpoint_path"),
            ("checkpoint_file_sha256", "reconstruction_checkpoint_file_sha256"),
            (
                "checkpoint_payload_sha256",
                "reconstruction_checkpoint_payload_sha256",
            ),
            ("plan_sha256", "reconstruction_plan_sha256"),
            ("input_hashes_sha256", "reconstruction_input_hashes_sha256"),
            ("statuses_sha256", "reconstruction_statuses_sha256"),
            ("evaluation_manifest_path", "evaluation_manifest_path"),
            (
                "evaluation_manifest_file_sha256",
                "evaluation_manifest_file_sha256",
            ),
            (
                "evaluation_manifest_payload_sha256",
                "evaluation_manifest_payload_sha256",
            ),
            ("evaluation_source_sha256", "evaluation_source_sha256"),
        ):
            value = f"{source_id}-{downstream_name}"
            downstream[f"downstream_{source_id}_{downstream_name}"] = value
            evaluation[f"{source_id}_{evaluation_name}"] = value

    selection._validate_revision_downstream_source_bindings(
        downstream,
        evaluation,
        ("v28", "v29"),
    )

    forged = dict(downstream)
    forged["downstream_v28_checkpoint_file_sha256"] = "forged"
    with pytest.raises(
        selection.SelectionAuthorityError,
        match="v28 downstream source differs",
    ):
        selection._validate_revision_downstream_source_bindings(
            forged,
            evaluation,
            ("v28", "v29"),
        )

    incomplete = dict(downstream)
    del incomplete["downstream_v29_statuses_sha256"]
    with pytest.raises(
        selection.SelectionAuthorityError,
        match="v29 downstream source differs",
    ):
        selection._validate_revision_downstream_source_bindings(
            incomplete,
            evaluation,
            ("v28", "v29"),
        )


def test_selection_blocks_if_count_score_manifest_binding_is_pending(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["count_score_manifest"] = {
        "status": "pending",
        "path": "artifacts/study/development/count_scores/manifest.json",
        "sha256": None,
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    schema_valid_payload = {
        "schema_version": 2,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "b" * 64,
        "retained_calibration_artifact_sha256": "c" * 64,
        "evaluation_manifest_sha256": "d" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "result_sha256": "e" * 64,
    }
    with pytest.raises(selection.SelectionAuthorityError, match="count-score.*pending"):
        selection._select_for_repository(
            schema_valid_payload,
            repository,
            require_clean=False,
        )


def test_schema_valid_count_score_manifest_cannot_change_the_frozen_config(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    manifest_path = (
        repository / "artifacts/study/development/count_scores/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["config_sha256"] = "0" * 64
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = selection._canonical_sha256(core)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["count_score_manifest"]["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match="configuration"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema_valid_calibration_cannot_invent_dataset_provenance(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    calibration_path = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["training"]["record_bindings"][0]["dataset_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in calibration.items() if key != "payload_sha256"
    }
    canonical = (
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    calibration["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["retained_calibration_artifact"]["sha256"] = hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match="score/dataset panel"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_cli_forwards_results_without_reconstructing_caller_design(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "select_development_candidate_script",
        Path("scripts/select_development_candidate.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sentinel = {
        "schema_version": 2,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "b" * 64,
        "retained_calibration_artifact_sha256": "c" * 64,
        "evaluation_manifest_sha256": "e" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "result_sha256": "d" * 64,
    }
    input_path = tmp_path / "selection-input.json"
    input_path.write_text(json.dumps(sentinel), encoding="utf-8")
    loaded = script._load(input_path)

    class Report:
        def to_dict(self):
            return {"selected": sentinel}

    monkeypatch.setattr(
        script,
        "select_development_candidate",
        lambda payload: Report() if payload is loaded else None,
    )

    assert script._report(loaded) == {"selected": sentinel}


def test_cli_main_recomputes_selection_against_repository_authority(
    tmp_path, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location(
        "select_development_candidate_script_main_authority",
        Path("scripts/select_development_candidate.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    input_path = tmp_path / "development_selection_input-downstream.json"
    output_path = tmp_path / "development_selection_report.json"
    sentinel = {"schema_version": 4}
    observed = []
    monkeypatch.setattr(script, "SELECTION_INPUT_PATH", input_path)
    monkeypatch.setattr(script, "SELECTION_REPORT_PATH", output_path)
    monkeypatch.setattr(
        script,
        "_secure_canonical_json",
        lambda path, label: (sentinel, "1" * 64)
        if path == input_path and label == "base selection-complete input"
        else pytest.fail("base selector read a noncanonical input path"),
    )
    monkeypatch.setattr(
        script,
        "_report",
        lambda payload, repository: observed.append((payload, repository))
        or {"selected_configuration": "v28-c01-nb-parent-c03"},
    )

    assert script.main([]) == 0
    assert observed == [(sentinel, script.REPOSITORY_ROOT)]
    assert json.loads(capsys.readouterr().out) == {
        "selected_configuration": "v28-c01-nb-parent-c03"
    }
    published = output_path.read_bytes()
    assert script.main([]) == 0
    capsys.readouterr()
    assert output_path.read_bytes() == published

    monkeypatch.setattr(
        script,
        "_report",
        lambda *_args: {"selected_configuration": "different"},
    )
    assert script.main([]) == 2
    assert output_path.read_bytes() == published


def test_base_selector_exposes_no_path_override_or_alternate_schema_four_input(
    tmp_path: Path,
) -> None:
    alternate = tmp_path / "alternate-schema-four.json"
    alternate.write_text(
        json.dumps({"schema_version": 4}),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/select_development_candidate.py",
            "--input",
            str(alternate),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "unrecognized arguments" in completed.stderr
    assert not output.exists()

    help_result = subprocess.run(
        [sys.executable, "scripts/select_development_candidate.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_result.returncode == 0
    assert "--input" not in help_result.stdout
    assert "--output" not in help_result.stdout


def test_base_selector_rejects_symlink_report_without_touching_referent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = importlib.util.spec_from_file_location(
        "select_development_candidate_symlink_report",
        Path("scripts/select_development_candidate.py"),
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    input_path = tmp_path / "evaluation/development_selection_input-downstream.json"
    report_path = tmp_path / "evaluation/development_selection_report.json"
    report_path.parent.mkdir(parents=True)
    referent = tmp_path / "referent.json"
    referent.write_bytes(b"unchanged\n")
    report_path.symlink_to(referent)
    monkeypatch.setattr(script, "SELECTION_INPUT_PATH", input_path)
    monkeypatch.setattr(script, "SELECTION_REPORT_PATH", report_path)
    monkeypatch.setattr(
        script,
        "_secure_canonical_json",
        lambda *_args: ({"schema_version": 4}, "1" * 64),
    )
    monkeypatch.setattr(
        script,
        "_report",
        lambda *_args: {"selected_configuration": "v27-c01-direct-r1-g1"},
    )

    assert script.main([]) == 2
    assert report_path.is_symlink()
    assert referent.read_bytes() == b"unchanged\n"


def test_base_selector_parent_swap_cannot_publish_report_in_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    specification = importlib.util.spec_from_file_location(
        "select_development_candidate_swapped_report",
        Path("scripts/select_development_candidate.py"),
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    parent = tmp_path / "evaluation"
    parent.mkdir()
    input_path = parent / "development_selection_input-downstream.json"
    report_path = parent / "development_selection_report.json"
    displaced = tmp_path / "evaluation-displaced"
    replacement = tmp_path / "evaluation-replacement"
    replacement.mkdir()
    monkeypatch.setattr(script, "SELECTION_INPUT_PATH", input_path)
    monkeypatch.setattr(script, "SELECTION_REPORT_PATH", report_path)
    monkeypatch.setattr(
        script,
        "_secure_canonical_json",
        lambda *_args: ({"schema_version": 4}, "1" * 64),
    )
    monkeypatch.setattr(
        script,
        "_report",
        lambda *_args: {"selected_configuration": "v27-c01-direct-r1-g1"},
    )
    real_link = promotion.os.link
    swapped = False

    def swap_parent(source_name, destination_name, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            parent.rename(displaced)
            replacement.rename(parent)
            swapped = True
        return real_link(source_name, destination_name, *args, **kwargs)

    monkeypatch.setattr(promotion.os, "link", swap_parent)

    assert script.main([]) == 2
    assert not os.path.lexists(report_path)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema_version":4, "records":[]}\n',
        b'{"schema_version":4,"schema_version":4}\n',
    ),
)
def test_base_selector_rejects_noncanonical_or_duplicate_key_fixed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
) -> None:
    specification = importlib.util.spec_from_file_location(
        "select_development_candidate_invalid_fixed_input",
        Path("scripts/select_development_candidate.py"),
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    input_path = tmp_path / "evaluation/development_selection_input-downstream.json"
    report_path = tmp_path / "evaluation/development_selection_report.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(raw)
    monkeypatch.setattr(script, "SELECTION_INPUT_PATH", input_path)
    monkeypatch.setattr(script, "SELECTION_REPORT_PATH", report_path)
    monkeypatch.setattr(
        script,
        "_report",
        lambda *_args: pytest.fail("invalid fixed input reached selection"),
    )

    assert script.main([]) == 2
    assert not report_path.exists()


def test_base_selector_rejects_symlinked_fixed_input_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = importlib.util.spec_from_file_location(
        "select_development_candidate_symlinked_fixed_input",
        Path("scripts/select_development_candidate.py"),
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    input_path = tmp_path / "evaluation/development_selection_input-downstream.json"
    report_path = tmp_path / "evaluation/development_selection_report.json"
    input_path.parent.mkdir(parents=True)
    referent = tmp_path / "input-referent.json"
    referent.write_bytes(b'{"schema_version":4}\n')
    input_path.symlink_to(referent)
    monkeypatch.setattr(script, "SELECTION_INPUT_PATH", input_path)
    monkeypatch.setattr(script, "SELECTION_REPORT_PATH", report_path)
    monkeypatch.setattr(
        script,
        "_report",
        lambda *_args: pytest.fail("symlinked fixed input reached selection"),
    )

    assert script.main([]) == 2
    assert not report_path.exists()


def test_base_selector_rejects_parent_swap_during_fixed_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    specification = importlib.util.spec_from_file_location(
        "select_development_candidate_swapped_fixed_input",
        Path("scripts/select_development_candidate.py"),
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    parent = tmp_path / "evaluation"
    parent.mkdir()
    input_path = parent / "development_selection_input-downstream.json"
    report_path = parent / "development_selection_report.json"
    input_path.write_bytes(b'{"schema_version":4}\n')
    displaced = tmp_path / "evaluation-displaced"
    replacement = tmp_path / "evaluation-replacement"
    replacement.mkdir()
    (replacement / input_path.name).write_bytes(b'{"schema_version":4}\n')
    monkeypatch.setattr(script, "SELECTION_INPUT_PATH", input_path)
    monkeypatch.setattr(script, "SELECTION_REPORT_PATH", report_path)
    monkeypatch.setattr(
        script,
        "_report",
        lambda *_args: pytest.fail("swapped fixed input reached selection"),
    )
    real_read = promotion.os.read
    swapped = False

    def swap_parent(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            parent.rename(displaced)
            replacement.rename(parent)
            swapped = True
        return real_read(descriptor, size)

    monkeypatch.setattr(promotion.os, "read", swap_parent)

    assert script.main([]) == 2
    assert not report_path.exists()


def test_cli_repository_selection_requires_clean_tracked_authority(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "select_development_candidate_script_clean_authority",
        Path("scripts/select_development_candidate.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    observed = []

    class Report:
        def to_dict(self):
            return {"trigger": "freeze_candidate"}

    monkeypatch.setattr(
        script,
        "_select_for_repository",
        lambda payload, repository, *, require_clean: observed.append(
            (payload, repository, require_clean)
        )
        or Report(),
    )
    payload = {"schema_version": 2}

    assert script._report(payload, tmp_path) == {"trigger": "freeze_candidate"}
    assert observed == [(payload, tmp_path, True)]


def test_cli_accepts_selection_complete_schema_four(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "select_development_candidate_schema_four_script",
        Path("scripts/select_development_candidate.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sentinel = {
        "schema_version": 4,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "b" * 64,
        "retained_calibration_artifact_sha256": "c" * 64,
        "evaluation_manifest_sha256": "e" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "revision_versions": [],
        "downstream_evidence": {
            "path": "artifacts/downstream",
            "manifest_file_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "plan_sha256": "3" * 64,
            "planned_denominator_count": 1,
            "endpoint_row_count": 8,
        },
        "result_sha256": "d" * 64,
    }
    input_path = tmp_path / "selection-input-schema-four.json"
    input_path.write_text(json.dumps(sentinel), encoding="utf-8")

    assert script._load(input_path) == sentinel


def test_cli_loaded_schema2_reaches_real_consumer_and_binds_manifest(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection

    spec = importlib.util.spec_from_file_location(
        "select_development_candidate_script_integration",
        Path("scripts/select_development_candidate.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    repository, _calibration_sha = _ready_repository(tmp_path)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Selection Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "study"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "authority"], check=True
    )
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        sources=_source_evidence(repository),
        reconstruction={},
        orthogonal={},
    )
    input_path = tmp_path / "schema2-selection-input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = script._load(input_path)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_reconstruction_evidence",
        lambda *_args: {"reconstruction_plan_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_orthogonal_evidence",
        lambda *_args: {"orthogonal_manifest_payload_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_evaluator_audits",
        lambda *_args: {"null_de_audits_sha256": "3" * 64},
    )

    report = script._report(loaded, repository)

    assert report["selected_configuration"] == "v27-c01-direct-r1-g1"
    assert (
        report["authority_bindings"]["evaluation_manifest_file_sha256"]
        == (payload["evaluation_manifest_sha256"])
    )
    assert (
        report["authority_bindings"]["source_ledger_file_sha256"]
        == (_source_evidence(repository)["ledger_file_sha256"])
    )


def test_result_payload_cannot_supply_attempts_declarations_or_design(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    payload = {
        "schema_version": 2,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "c" * 64,
        "retained_calibration_artifact_sha256": "d" * 64,
        "evaluation_manifest_sha256": "e" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "attempts": [],
        "declarations": [],
        "design": {},
        "result_sha256": "b" * 64,
    }

    with pytest.raises(selection.SelectionAuthorityError, match="missing or extra"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_consumer_requires_fixed_evaluation_manifest(tmp_path, monkeypatch):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload_core = {
        key: value for key, value in payload.items() if key != "result_sha256"
    }
    payload_core["evaluation_manifest_sha256"] = "e" * 64
    payload = {
        **payload_core,
        "result_sha256": selection._canonical_sha256(payload_core),
    }
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(
        selection.SelectionAuthorityError, match="evaluation manifest.*absent"
    ):
        selection._select_for_repository(payload, repository, require_clean=False)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    (
        (
            lambda evaluation: evaluation.__setitem__("manifest_sha256", "0" * 64),
            "evaluation manifest payload checksum",
        ),
        (
            lambda evaluation: (
                evaluation.__setitem__("selection_evidence_sha256", "0" * 64),
                evaluation.__setitem__(
                    "manifest_sha256",
                    __import__(
                        "maskimpute_benchmark.selection", fromlist=["_canonical_sha256"]
                    )._canonical_sha256(
                        {
                            key: value
                            for key, value in evaluation.items()
                            if key != "manifest_sha256"
                        }
                    ),
                ),
            ),
            "selection evidence checksum",
        ),
    ),
)
def test_schema2_consumer_rejects_evaluation_envelope_tampering(
    tmp_path, monkeypatch, corrupt, message
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload = _attach_evaluation_manifest(repository, payload, corrupt=corrupt)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match=message):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_consumer_revalidates_source_evidence(tmp_path, monkeypatch):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload = _attach_evaluation_manifest(repository, payload)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match="source evidence"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_consumer_rejects_changed_bound_source_bytes(tmp_path, monkeypatch):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    sources = _source_evidence(repository)
    payload = _attach_evaluation_manifest(repository, payload, sources=sources)
    first_artifact = repository / sources["artifacts"][0]["path"]
    first_artifact.write_bytes(b"tampered")
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match="source evidence"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_consumer_requires_bound_reconstruction_checkpoint(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload = _attach_evaluation_manifest(
        repository, payload, sources=_source_evidence(repository)
    )
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(
        selection.SelectionAuthorityError, match="reconstruction.*checkpoint"
    ):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_consumer_rejects_reconstruction_input_authority_drift(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        sources=_source_evidence(repository),
        reconstruction=_minimal_reconstruction_evidence(repository),
    )
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match="plan/input authority"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_consumer_rebuilds_reconstruction_plan_authority(tmp_path, monkeypatch):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    reconstruction = _minimal_reconstruction_evidence(
        repository,
        input_hashes=_reconstruction_inputs(authority, status, payload),
    )
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        sources=_source_evidence(repository),
        reconstruction=reconstruction,
    )
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(
        selection.SelectionAuthorityError, match="reconstruction plan authority"
    ):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_consumer_rejects_reconstruction_raw_artifact_alias(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.development_evaluation as development_evaluation
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    inputs = _reconstruction_inputs(authority, status, payload)
    reconstruction = _minimal_reconstruction_evidence(repository, input_hashes=inputs)
    reconstruction["raw_artifacts"] = [
        {
            "run_id": "run-forged",
            "kind": "stdout",
            "path": "artifacts/study/development/competition-reconstruction/checkpoint.json",
            "file_sha256": reconstruction["checkpoint_file_sha256"],
        }
    ]
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        sources=_source_evidence(repository),
        reconstruction=reconstruction,
    )
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    monkeypatch.setattr(
        evaluation,
        "_rebuild_reconstruction_plan",
        lambda *_args: SimpleNamespace(
            input_hashes=inputs, plan_sha256=reconstruction["plan_sha256"]
        ),
    )
    prepared = {}
    monkeypatch.setattr(
        evaluation,
        "_prepare_reconstruction_datasets",
        lambda *_args: prepared,
    )

    def load_reconstruction_checkpoint(*_args, prepared_datasets):
        assert prepared_datasets is prepared
        return SimpleNamespace(
            checkpoint_file_sha256=reconstruction["checkpoint_file_sha256"],
            checkpoint_sha256=reconstruction["checkpoint_sha256"],
            plan_sha256=reconstruction["plan_sha256"],
            input_hashes=inputs,
            raw_artifacts=(),
        )

    monkeypatch.setattr(
        development_evaluation,
        "load_completed_reconstruction_checkpoint",
        load_reconstruction_checkpoint,
    )

    with pytest.raises(
        selection.SelectionAuthorityError, match="raw artifact denominator"
    ):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_rehash_all_cannot_change_reconstructed_efficacy_metric(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.development_evaluation as development_evaluation
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    independently_rebuilt_records = json.loads(json.dumps(payload["records"]))
    efficacy = next(row for row in payload["records"] if row["metric"] == "mse")
    efficacy["value"] = float(efficacy["value"]) + 0.25
    inputs = _reconstruction_inputs(authority, status, payload)
    reconstruction = _minimal_reconstruction_evidence(repository, input_hashes=inputs)
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        reconstruction=reconstruction,
    )
    plan = SimpleNamespace(
        input_hashes=inputs,
        plan_sha256=reconstruction["plan_sha256"],
    )
    evidence = SimpleNamespace(
        checkpoint_file_sha256=reconstruction["checkpoint_file_sha256"],
        checkpoint_sha256=reconstruction["checkpoint_sha256"],
        plan_sha256=reconstruction["plan_sha256"],
        input_hashes=inputs,
        raw_artifacts=(),
    )
    monkeypatch.setattr(
        evaluation, "_validate_evaluation_source_evidence", lambda *_: {}
    )
    monkeypatch.setattr(evaluation, "_validate_orthogonal_evidence", lambda *_: {})
    monkeypatch.setattr(evaluation, "_validate_evaluator_audits", lambda *_: {})
    monkeypatch.setattr(evaluation, "_rebuild_reconstruction_plan", lambda *_: plan)
    prepared = {}
    monkeypatch.setattr(
        evaluation,
        "_prepare_reconstruction_datasets",
        lambda *_: prepared,
        raising=False,
    )

    def load_reconstruction_checkpoint(*_args, prepared_datasets):
        assert prepared_datasets is prepared
        return evidence

    monkeypatch.setattr(
        development_evaluation,
        "load_completed_reconstruction_checkpoint",
        load_reconstruction_checkpoint,
    )
    monkeypatch.setattr(
        development_evaluation,
        "build_reconstruction_selection_records",
        lambda *_args, **_kwargs: SimpleNamespace(
            records=tuple(independently_rebuilt_records),
            null_de_audits=(),
        ),
    )

    with pytest.raises(
        evaluation.EvaluationManifestError,
        match="reconstructed selection records",
    ):
        evaluation.validate_selection_evaluation_manifest(
            repository, payload, authority, status
        )


def test_schema2_consumer_requires_bound_orthogonal_outputs(tmp_path, monkeypatch):
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        sources=_source_evidence(repository),
        reconstruction={},
    )
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_reconstruction_evidence",
        lambda *_args: {},
    )

    with pytest.raises(selection.SelectionAuthorityError, match="orthogonal.*binding"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_orthogonal_evidence_revalidates_every_output_byte(tmp_path):
    import maskimpute_benchmark.evaluation_manifest as evaluation

    repository = tmp_path / "repository"
    repository.mkdir()
    orthogonal, output_path = _minimal_orthogonal_evidence(repository)
    output_path.write_bytes(b"tampered")

    with pytest.raises(
        evaluation.EvaluationManifestError, match="orthogonal output.*checksum"
    ):
        evaluation._validate_orthogonal_evidence(repository, orthogonal)


def test_schema2_consumer_requires_current_orthogonal_authority(tmp_path, monkeypatch):
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    orthogonal, _output_path = _minimal_orthogonal_evidence(repository)
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        sources=_source_evidence(repository),
        reconstruction={},
        orthogonal=orthogonal,
    )
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_reconstruction_evidence",
        lambda *_args: {},
    )

    with pytest.raises(selection.SelectionAuthorityError, match="orthogonal authority"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema2_rehash_all_cannot_change_orthogonal_interval_and_audit(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.development_evaluation as development_evaluation
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection
    from maskimpute_benchmark.runner import derive_authorized_configurations

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    independently_recomputed_intervals = json.loads(
        json.dumps(payload["orthogonal_intervals"])
    )
    independently_recomputed_audits = [
        {
            **row,
            "reason": None,
            "n_biological_units": 2,
            "n_technical_units": 4,
            "n_boot": 10_000,
            "bootstrap_sha256": "6" * 64,
            "aggregation": "fixed_evaluator_aggregation",
            "inference_scope": "fixed_evaluator_scope",
            "profile_scale": "fixed_evaluator_scale",
        }
        for row in independently_recomputed_intervals
    ]
    payload["orthogonal_intervals"][0]["estimate"] = 0.25
    tampered_audits = json.loads(json.dumps(independently_recomputed_audits))
    tampered_audits[0]["estimate"] = 0.25

    ledger = json.loads(
        (repository / "study/development_search.json").read_text(encoding="utf-8")
    )
    configurations = derive_authorized_configurations(
        ledger["configurations"],
        authority.ablation_specs,
        authority.method_bindings,
    )
    orthogonal_authority = {
        "inputs": [
            {
                "source_id": source_id,
                "source_dataset_sha256": "1" * 64,
                "method_input_sha256": "2" * 64,
                "shape": [2, 2],
                "cell_ids_sha256": "3" * 64,
                "gene_ids_sha256": "4" * 64,
            }
            for source_id in (
                "cite-seq-cbmc-rna-protein",
                "tung-ipsc-ercc-bulk-replicates",
            )
        ],
        "configurations": [
            {
                "configuration_id": value.configuration_id,
                "configuration_sha256": value.configuration_sha256,
                "payload": dict(value.payload),
            }
            for value in configurations
            if value.method_id == "maskimpute" and value.kind == "candidate_search"
        ],
        "model_seeds": [42, 43, 44],
        "artifact_bindings": {
            "count_model_config_sha256": authority.count_model_config_sha256,
            "retained_calibration_artifact_sha256": (
                authority.retained_calibration.sha256
            ),
            "score_fit_policy": (
                "refit_cross_fitted_count_score_from_truth_free_input"
            ),
        },
    }
    from maskimpute_benchmark.selection import _canonical_sha256

    orthogonal_core = {
        "schema_version": 1,
        "artifact_type": "maskimpute_orthogonal_method_outputs",
        "authority": orthogonal_authority,
        "status": "completed",
        "planned_record_count": 0,
        "records": [],
    }
    orthogonal_manifest = {
        **orthogonal_core,
        "manifest_sha256": _canonical_sha256(orthogonal_core),
    }
    orthogonal_path = repository / (
        "artifacts/study/development/evaluation/orthogonal/orthogonal_outputs.json"
    )
    orthogonal_path.parent.mkdir(parents=True, exist_ok=True)
    orthogonal_path.write_text(
        json.dumps(orthogonal_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    orthogonal = {
        "manifest_path": str(orthogonal_path.relative_to(repository)),
        "manifest_file_sha256": hashlib.sha256(
            orthogonal_path.read_bytes()
        ).hexdigest(),
        "manifest_sha256": orthogonal_manifest["manifest_sha256"],
        "records": [],
    }
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        orthogonal=orthogonal,
        orthogonal_audits=tampered_audits,
    )
    output_evidence = SimpleNamespace(
        manifest_file_sha256=orthogonal["manifest_file_sha256"],
        manifest_sha256=orthogonal["manifest_sha256"],
        records=(),
    )
    panel = SimpleNamespace(method_inputs=(), cite=object(), tung=object())
    monkeypatch.setattr(
        evaluation, "_validate_evaluation_source_evidence", lambda *_: {}
    )
    monkeypatch.setattr(evaluation, "_validate_reconstruction_evidence", lambda *_: {})
    monkeypatch.setattr(evaluation, "_validate_evaluator_audits", lambda *_: {})
    monkeypatch.setattr(
        development_evaluation,
        "load_orthogonal_output_evidence",
        lambda *_args, **_kwargs: output_evidence,
    )
    monkeypatch.setattr(
        development_evaluation,
        "prepare_real_orthogonal_panel",
        lambda *_: panel,
    )
    monkeypatch.setattr(
        development_evaluation,
        "_orthogonal_authority_core",
        lambda *_args, **_kwargs: orthogonal_authority,
    )
    monkeypatch.setattr(
        development_evaluation,
        "evaluate_real_orthogonal_intervals",
        lambda *_args, **_kwargs: SimpleNamespace(
            intervals=tuple(independently_recomputed_intervals),
            audits=tuple(independently_recomputed_audits),
        ),
    )

    with pytest.raises(
        evaluation.EvaluationManifestError,
        match="orthogonal intervals.*independently recomputed",
    ):
        evaluation.validate_selection_evaluation_manifest(
            repository, payload, authority, status
        )


def test_schema2_consumer_requires_complete_evaluator_audits(tmp_path, monkeypatch):
    import maskimpute_benchmark.evaluation_manifest as evaluation
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    payload = _attach_evaluation_manifest(
        repository,
        payload,
        sources=_source_evidence(repository),
        reconstruction={},
        orthogonal={},
    )
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_reconstruction_evidence",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_orthogonal_evidence",
        lambda *_args: {},
    )

    with pytest.raises(
        selection.SelectionAuthorityError, match="null-DE audit denominator"
    ):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_evaluator_audits_cannot_change_selection_values(tmp_path):
    import maskimpute_benchmark.evaluation_manifest as evaluation

    repository = tmp_path / "repository"
    checkpoint_path = (
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    checkpoint_path.parent.mkdir(parents=True)
    run = {
        "run_id": "run-observed-test",
        "method_id": "observed",
        "configuration_id": "registry-default",
        "configuration_kind": "registry",
        "mechanism": "symsim",
        "biological_id": "draw-01",
        "technical_view": "moderate",
        "dataset_id": "dataset-test",
        "model_seed": None,
        "evaluator_output_file_sha256": "4" * 64,
    }
    checkpoint = {"records": [{"run": run}]}
    checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")) + "\n"
    )
    checkpoint_sha256 = "5" * 64
    entropy = hashlib.sha256()
    entropy.update(b"maskimpute-null-de-post-execution-entropy-v1\0")
    entropy.update(checkpoint_sha256.encode("ascii"))
    entropy.update(b"\0symsim\0draw-01")
    null_record = {
        "mechanism": "symsim",
        "biological_id": "draw-01",
        "technical_view": "moderate",
        "dataset_id": "dataset-test",
        "method": "observed",
        "model_seed": None,
        "metric": "null_de_fpr",
        "value": 0.05,
        "status": "completed",
    }
    interval = {
        "configuration": "v27-c01-direct-r1-g1",
        "endpoint": "rna_protein_concordance",
        "comparison": "observed",
        "estimate": 0.0,
        "ci_lower": -0.01,
        "ci_upper": 0.01,
        "status": "completed",
    }
    manifest = {
        "reconstruction": {
            "checkpoint_path": (
                "artifacts/study/development/competition-reconstruction/checkpoint.json"
            ),
            "checkpoint_file_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "null_de_audits": [
            {
                "run_id": run["run_id"],
                "dataset_id": run["dataset_id"],
                "method": "observed",
                "model_seed": None,
                "status": "completed",
                "value": 0.10,
                "nominal_alpha": 0.05,
                "n_tested_genes": 100,
                "fixed_gene_count": 100,
                "split_entropy_sha256": entropy.hexdigest(),
                "split_entropy_derivation": (
                    "sha256(completed_checkpoint_sha256,mechanism,biological_id)"
                ),
                "split_sha256": "6" * 64,
                "gene_mask_sha256": "7" * 64,
                "reason": None,
                "evaluator_output_file_sha256": "4" * 64,
            }
        ],
        "orthogonal_audits": [
            {
                **interval,
                "reason": None,
                "n_biological_units": 1,
                "n_technical_units": 4,
                "n_boot": 100,
                "bootstrap_sha256": "8" * 64,
                "aggregation": "paired_cell_bootstrap",
                "inference_scope": "single_specimen",
                "profile_scale": "matched_marker_rank_correlation_across_cells",
            }
        ],
    }
    data = {"records": [null_record], "orthogonal_intervals": [interval]}
    authority = SimpleNamespace(
        attempts=(), declarations=(SimpleNamespace(id="observed"),)
    )

    with pytest.raises(
        evaluation.EvaluationManifestError,
        match="null-DE audit differs from its selection record",
    ):
        evaluation._validate_evaluator_audits(repository, manifest, data, authority)


def test_result_payload_checksum_is_verified_before_dataset_access(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    payload = {
        "schema_version": 2,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "c" * 64,
        "retained_calibration_artifact_sha256": "d" * 64,
        "evaluation_manifest_sha256": "e" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "result_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: pytest.fail("dataset status should not be read"),
    )

    with pytest.raises(selection.SelectionAuthorityError, match="result checksum"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_authority_loader_rejects_duplicate_json_keys(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    contract = repository / "study/selection_contract.json"
    text = contract.read_text(encoding="utf-8")
    contract.write_text(
        text.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(selection.SelectionAuthorityError, match="duplicate JSON key"):
        selection._load_selection_authority(repository, require_clean=False)


def test_authority_must_be_tracked_and_clean_for_public_selection(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Selection Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "study"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "authority"], check=True
    )

    selection._load_selection_authority(repository, require_clean=True)
    contract = repository / "study/selection_contract.json"
    contract.write_text(contract.read_text(encoding="utf-8") + "\n")

    with pytest.raises(selection.SelectionAuthorityError, match="differs"):
        selection._load_selection_authority(repository, require_clean=True)


def test_finalizer_validates_both_artifacts_before_atomically_marking_ready(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    ready_authority = selection._load_selection_authority(
        repository, require_clean=False
    )
    status, _payload = _status_and_payload(ready_authority)
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for field in ("count_score_manifest", "retained_calibration_artifact"):
        ledger[field]["status"] = "pending"
        ledger[field]["sha256"] = None
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    score_path = repository / "artifacts/study/development/count_scores/manifest.json"
    calibration_path = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    monkeypatch.setattr(
        selection,
        "_revalidate_development_score_preparation",
        lambda _repository: {
            "status": "reused",
            "count_score_manifest_file_sha256": hashlib.sha256(
                score_path.read_bytes()
            ).hexdigest(),
            "calibration_file_sha256": hashlib.sha256(
                calibration_path.read_bytes()
            ).hexdigest(),
        },
    )

    finalized = selection._finalize_development_artifact_bindings_for_repository(
        repository,
        require_clean=False,
    )

    updated = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert updated["count_score_manifest"]["status"] == "ready"
    assert updated["retained_calibration_artifact"]["status"] == "ready"
    assert (
        updated["count_score_manifest"]["sha256"]
        == finalized["count_score_manifest_sha256"]
    )
    assert (
        updated["retained_calibration_artifact"]["sha256"]
        == finalized["retained_calibration_artifact_sha256"]
    )
    assert finalized["next_required_action"] == "commit_development_search_ledger"


def test_public_finalizer_has_no_caller_controlled_paths_or_hashes():
    from maskimpute_benchmark.selection import (
        finalize_development_artifact_bindings,
    )

    assert (
        tuple(inspect.signature(finalize_development_artifact_bindings).parameters)
        == ()
    )


def test_finalization_cli_exposes_no_path_or_hash_arguments(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "finalize_development_authority_script",
        Path("scripts/finalize_development_authority.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(
        script,
        "finalize_development_artifact_bindings",
        lambda: {"next_required_action": "commit_development_search_ledger"},
    )

    assert script._finalize() == {
        "next_required_action": "commit_development_search_ledger"
    }
