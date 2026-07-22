# Fair Comparator Tuning Implementation Plan

## Governing direct-identity amendment (2026-07-20)

The approved direct-identity design at
`docs/superpowers/specs/2026-07-20-direct-identity-fair-comparator-design.md`
and its implementation plan at
`docs/superpowers/plans/2026-07-20-direct-identity-fair-comparator.md` govern
the fair-comparator segment of this plan. Tasks 8--18 below retain all of their
scientific grid, denominator, budget, readiness, selection, propagation,
documentation, and verification requirements, but their identity and artifact
mechanisms must use the following substitutions:

| Superseded mechanism | Required direct replacement |
| --- | --- |
| configuration or payload content summary | full typed payload plus configuration ID |
| comparator authority file/payload summary | path, schema version, and `authority_revision`, followed by full authority validation |
| method/source/configuration identity summary | `ComparatorMethodBinding` compared field by field |
| request or entry summary | complete `ComparatorRunIdentity` |
| plan summary and input-summary map | complete ordered plan snapshot plus `PreparedInputDescriptor` values |
| stdout/stderr summary | canonical stream name, original byte count, capture policy, and terminal reason |
| smoke/selection receipt summary | full canonical receipt bytes and recomputed derived values |
| create-only concurrent equality through summaries | byte-for-byte comparison of complete canonical artifacts |

Every digest-oriented fair-comparator code snippet or field in the older Tasks
8--18 text is superseded. Implementers must remove it rather than reproduce,
rename, alias, or dual-write it. Every later Task 8--18 brief must include both
governing paths above, the direct plan's Global Constraints, and this
substitution table before work begins. Reviews apply the direct requirements
first and preserve the older task text only as the binding scientific contract.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace default-only comparator evaluation with a bounded, development-only 34-configuration tuning competition whose selected comparator identities propagate unchanged through candidate selection, revision assessment, publication freeze, final evaluation, trajectory evaluation, scaling, analysis, and the Genome Biology manuscript.

**Architecture:** A new cycle-free `comparator_tuning` module owns the strict tracked authority, typed adapter decoding, immutable configuration identities, comparator-only collapse/selection, readiness, smoke evidence, and receipt validation. The runner owns the 2,896-entry execution plan, budgets, storage, dispatch, and checkpoint evidence; downstream modules consume only the validated selected-configuration projection and never reopen tuning or reconstruct adapter defaults. Every descendant binds both the tuning authority and the create-only comparator-selection receipt, while unavailable stochastic methods retain their prespecified seeded structural denominator with a frozen nonexecution identity.

**Tech Stack:** Python 3.12, NumPy 2, frozen dataclasses, canonical JSON and SHA-256 domain separation, POSIX no-follow/create-only publication, zlib bounds, pytest 8, Ruff, LaTeX.

## Global Constraints

- The governing design is `docs/superpowers/specs/2026-07-18-fair-comparator-tuning-design.md`; calibration repair, fresh scientific execution, publication-asset export, numerical Results, and licensing/release remain separate subprojects.
- Do not run or resume any development or final scientific competition in this implementation plan. The current calibration contradiction continues to block scientific execution even after these code paths are complete.
- The tracked comparator grid is exactly 34 configurations in the approved method/configuration order: eight four-point grids plus the sole ALRA and SAVER defaults.
- Development planning is exactly 2,896 rows: 16 observed, 48 capacity control, 1,200 MaskImpute search/ablation, and 1,632 comparator-tuning rows.
- Each comparator method attempts one complete 16-dataset by 3-seed configuration block before the next configuration; its upstream default block is first.
- The scheduled same-input denominator is exactly `observed`, `capacity-matched-ae`, `alra`, `magic`, `dca`, `scvi`, `saver`, `scziva`, `afmf`, `biaeimpute`, `sccr`, `scsdae` in registry order.
- Controls `observed` and `capacity-matched-ae` must be complete; all of `alra`, `magic`, `dca`, `scvi`, and `saver` must be selectable; at least three of `scziva`, `afmf`, `biaeimpute`, and `sccr` must be selectable. scSDAE is always scheduled and numerical when selectable.
- Comparator selection uses only that method's development results. MaskImpute values, final data, downstream endpoints, and external-reference evidence cannot affect comparator eligibility, Pareto membership, ranks, or selected IDs.
- Collapse order is seeds, paired technical views, then biological draws. Five panel-wide metrics retain eight biological units; `mse_pre_dropout_zero` retains the two SymSim biological units.
- Pareto filtering uses all six configuration medians. Average ranks are computed only within the Pareto set and serialized as exact integer quarter-ranks; the complete approved integer tuple is the only selection rule.
- `failed`, `timeout`, `resource_exceeded`, and `unavailable` are intrinsic terminal outcomes. `budget_exhausted`, `blocked_authority`, and any persisted `infrastructure_error` block selection and require new authority plus a fresh checkpoint.
- Configurations, time, and failures remain in the scheduled denominator. No execution-order subset may become selection-eligible.
- Selected comparator payloads are full adapter payloads. Unknown/missing fields, bool-as-int, type coercion, nonfinite values, partial overrides, registry-default comparator fallback, and post-decode payload drift fail closed.
- The final structural denominator is always 1,760 and the trajectory denominator is always 44. An unavailable stochastic same-input method changes action and identity for exactly 120 final or three trajectory rows; it never collapses their seeds.
- Revision v28/v29 plans execute only newly authorized MaskImpute candidates and reuse base controls plus selected comparator records byte-for-byte.
- Output publication follows existing owned regular-file, no-symlink, no-special-file, create-only, atomic, concurrent-identical, and immediate-revalidation conventions.
- Raw stderr and absolute private paths are never retained in a publication artifact. Development log artifacts retain only canonical stream name, byte count, and SHA-256 of the original bytes.
- Test first: every production change begins with the named failing regression, is implemented minimally, and is committed only after focused and adjacent tests pass.

### Narrow design reconciliations required by repository arithmetic

The implementation must add an explicit erratum to the governing design before production code is merged. Literal reuse of scaling's 64 MiB-per-log and 2 MiB-per-receipt ceilings would reserve about 395 GiB for 2,896 attempts, contradicting the approved 20--27 GiB development estimate. Use these development-specific, tracked ceilings instead:

```python
DEVELOPMENT_MAX_LOG_RECEIPT_BYTES = 64 * 1024
DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES = 64 * 1024
DEVELOPMENT_MAX_RECORD_BYTES = 64 * 1024
DEVELOPMENT_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
DEVELOPMENT_STORAGE_RESERVE_BYTES = 1024**3
```

For the maximum approved 900-by-500 retained shape, the fail-closed bound is approximately 25.21 GiB: two raw little-endian float64 matrices for every executable row, the zlib bound for 1,200 MaskImpute score matrices, two log receipts plus one executor receipt and one record allowance per row, one 64 MiB checkpoint allowance, and the shared 1 GiB reserve. Available space is checked before any output directory or scientific artifact is created; existing evidence is never removed or relocated.

The fixed smoke policy is one deterministic truth-free 900-by-500 raw-count `MethodInput`, generated from the tracked integer formula `(17 * cell + 31 * gene + 7 * (cell ^ gene)) % 6`, with batch labels alternating `batch-0`/`batch-1`, model seed 42, and all 34 configurations. No evaluator target, score, metric, native matrix, or imputed matrix is retained. For each method, `48 * sum(one-smoke-runtime-per-configuration)` must not exceed its 8-hour GPU or 24-hour CPU budget, and every peak must respect 48 GiB RAM and 14 GiB GPU. The create-only receipt retains only configuration identities, terminal status/reason, runtime, resource peaks/provenance, the projection multiplier 48, and hashes. Every configuration must complete; otherwise the tracked grid or budget requires a reviewed amendment before scientific execution.

---

## File Structure and Responsibility Map

### New files

- `maskimpute_benchmark/comparator_tuning.py`: strict authority parsing, full-payload typed decoding/encoding, identity binding, smoke receipt creation/validation, seed-view-draw collapse, Pareto/rank selection, readiness, create-only selection receipt publication.
- `study/comparator_tuning.json`: sole tracked 34-configuration authority, selection/readiness policy, smoke policy, budgets, storage ceilings, and output paths.
- `scripts/run_comparator_tuning_smoke.py`: no-override fixed smoke entry point.
- `scripts/select_comparator_configurations.py`: no-override fixed comparator selector.
- `tests/test_comparator_tuning.py`: pure and repository-bound authority, decoder, identity, smoke, selection, readiness, receipt, and tamper tests.

### Existing files modified by responsibility

- `study/selection_contract.json`, `study/development_search.json`: bind tuning authority and replace the overloaded comparator list with four exact sets.
- `maskimpute_benchmark/selection.py`: validate new authority chain and consume only the receipt-projected comparison population.
- `maskimpute_benchmark/runner.py`: carry bound identities, expand/order 2,896 rows, enforce budgets/storage/log receipts, typed dispatch, candidate-only revisions, and exact checkpoint replay.
- `maskimpute_benchmark/development_evaluation.py`, `maskimpute_benchmark/evaluation_manifest.py`: validate the fixed receipt and project only selected comparator configurations.
- `maskimpute_benchmark/revisions.py`, `maskimpute_benchmark/revision_evaluation.py`: preserve comparator receipt bindings and reuse base comparator/control evidence.
- `maskimpute_benchmark/publication_freeze.py`: freeze complete comparator sets, selection receipt, selected payloads/identities, and unavailable nonexecution identities.
- `maskimpute_benchmark/final_runner.py`: derive final/trajectory configurations only from the frozen map and preserve seeded nonexecutions.
- `maskimpute_benchmark/scaling.py`: use exact frozen MAGIC/DCA/scVI payloads and identities.
- `maskimpute_benchmark/downstream_evidence.py`, `maskimpute_benchmark/final_analysis.py`, `maskimpute_benchmark/publication_synthesis.py`: replay new identity fields, distinguish scheduled and numerical denominators, and block unsupported superiority claims.
- `docs/development-selection-workflow.md`, `paper/manuscript.tex`, `paper/submission_checklist.md`, `docs/genome-biology-submission-checklist.md`: explain the fair development-only comparator selection and complete execution-status denominator without provisional results.
- Focused test files are listed task by task; repository hygiene tests close generated-artifact leakage.

---

### Task 1: Ratify exact authority, storage, and smoke contracts

**Files:**

- Modify: `docs/superpowers/specs/2026-07-18-fair-comparator-tuning-design.md`
- Create: `study/comparator_tuning.json`
- Create: `maskimpute_benchmark/comparator_tuning.py`
- Create: `tests/test_comparator_tuning.py`

**Interfaces:**

- Consumes: `MethodRegistry`, the ten adapter configuration dataclasses, and `canonical_sha256(value: object) -> str`.
- Produces: `ComparatorConfiguration`, `ComparatorTuningAuthority`, `parse_comparator_tuning_authority(payload, *, registry, file_sha256)`, `load_comparator_tuning_authority(repository, *, registry, require_clean=True)`, and exact tracked storage/smoke constants.

- [ ] **Step 1: Amend the approved design with the exact reconciliation**

Append this subsection immediately before `## Verification requirements`:

```markdown
## Development storage and smoke erratum

The development runner uses development-specific retained-artifact ceilings:
64 KiB for each hash-only stdout/stderr receipt, 64 KiB for the canonical
executor receipt, 64 KiB per planned JSON record, 64 MiB for one replacement
checkpoint, and a shared 1 GiB reserve. Scaling's larger interactive log and
receipt ceilings do not apply to this 2,896-row retained checkpoint. At the
maximum 900-by-500 retained shape the exact fail-closed bound is approximately
25.21 GiB. The preflight counts two raw float64 matrices for every executable
row and the zlib compression bound for every applicable MaskImpute score matrix.

The fixed nonstudy smoke input is one truth-free 900-by-500 raw-count matrix
whose entry at `(cell, gene)` is
`(17 * cell + 31 * gene + 7 * (cell ^ gene)) % 6`, with alternating two-batch
labels. All thirty-four configurations run once at model seed 42 without an
evaluator. For each method, the sum of its configuration runtimes is multiplied
by 48 and must fit the tracked 8-hour GPU or 24-hour CPU budget; every measured
peak must fit 48 GiB RAM and 14 GiB GPU. The receipt retains only status,
reason, time/resource provenance, identities, and hashes; native/imputed outputs
are discarded. Any noncompleted smoke row requires a reviewed pre-study
amendment. Scientific execution requires this validated create-only receipt.
```

- [ ] **Step 2: Write the failing exact-grid test**

Add these imports and test to `tests/test_comparator_tuning.py`:

```python
from dataclasses import asdict
from pathlib import Path

from maskimpute_benchmark.comparator_tuning import (
    DEVELOPMENT_MAX_CHECKPOINT_BYTES,
    DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
    DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
    DEVELOPMENT_MAX_RECORD_BYTES,
    DEVELOPMENT_STORAGE_RESERVE_BYTES,
    encode_comparator_configuration,
    load_comparator_tuning_authority,
)
from maskimpute_benchmark.methods import load_method_registry


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ORDER = {
    "alra": ("alra-default",),
    "magic": ("magic-t03", "magic-t01", "magic-t05", "magic-t07"),
    "dca": ("dca-h64-32-64", "dca-h32-16-32", "dca-h32-32", "dca-h64-64"),
    "scvi": ("scvi-z10", "scvi-z05", "scvi-z20", "scvi-z30"),
    "saver": ("saver-default",),
    "scziva": ("scziva-tau-0p001", "scziva-tau-0p0001", "scziva-tau-0p01", "scziva-tau-0p05"),
    "afmf": ("afmf-sigma-3", "afmf-sigma-1", "afmf-sigma-2", "afmf-sigma-4"),
    "biaeimpute": ("biaeimpute-z128", "biaeimpute-z32", "biaeimpute-z64", "biaeimpute-z256"),
    "sccr": ("sccr-k15", "sccr-k05", "sccr-k10", "sccr-k30"),
    "scsdae": ("scsdae-zero-1", "scsdae-zero-0p25", "scsdae-zero-0p5", "scsdae-zero-0p75"),
}


def test_tracked_authority_has_exact_grid_and_operational_contract() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )

    assert authority.method_order == tuple(EXPECTED_ORDER)
    assert len(authority.configurations) == 34
    assert {
        method_id: tuple(
            row.configuration_id
            for row in authority.configurations_for(method_id)
        )
        for method_id in authority.method_order
    } == EXPECTED_ORDER
    assert all(
        sum(row.is_upstream_default for row in authority.configurations_for(method_id))
        == 1
        and authority.configurations_for(method_id)[0].is_upstream_default
        for method_id in authority.method_order
    )
    assert authority.scheduled_same_input_ids == (
        "observed", "capacity-matched-ae", "alra", "magic", "dca", "scvi",
        "saver", "scziva", "afmf", "biaeimpute", "sccr", "scsdae",
    )
    assert authority.required_control_ids == ("observed", "capacity-matched-ae")
    assert authority.established_comparator_ids == ("alra", "magic", "dca", "scvi", "saver")
    assert authority.modern_core_ids == ("scziva", "afmf", "biaeimpute", "sccr")
    assert authority.model_seeds == (42, 43, 44)
    assert authority.receipt_path == "artifacts/study/development/evaluation/comparator_selection.json"
    assert authority.smoke_receipt_path == "artifacts/study/development/evaluation/comparator_smoke.json"
    assert (
        DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
        DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
        DEVELOPMENT_MAX_RECORD_BYTES,
        DEVELOPMENT_MAX_CHECKPOINT_BYTES,
        DEVELOPMENT_STORAGE_RESERVE_BYTES,
    ) == (65_536, 65_536, 65_536, 67_108_864, 1_073_741_824)
    for row in authority.configurations:
        assert row.payload_sha256 == row.observed_payload_sha256
        assert encode_comparator_configuration(row.decode()) == dict(row.payload)
```

- [ ] **Step 3: Run the test and verify the intended failure**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py::test_tracked_authority_has_exact_grid_and_operational_contract \
  -q -W error -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError: No module named 'maskimpute_benchmark.comparator_tuning'`.

- [ ] **Step 4: Add the immutable authority types and exact parser**

Create `maskimpute_benchmark/comparator_tuning.py` with these public definitions and parser skeleton; the helper names shown here are the names later tasks must use:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .methods import (
    AFMFConfig, ALRAConfig, BiAEImputeConfig, DCAConfig, MAGICConfig,
    SAVERConfig, SCCRConfig, SCSDaeConfig, SCVIConfig, SCZivaConfig,
)
from .methods import MethodRegistry
from .protocol import canonical_sha256


DEVELOPMENT_MAX_LOG_RECEIPT_BYTES = 64 * 1024
DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES = 64 * 1024
DEVELOPMENT_MAX_RECORD_BYTES = 64 * 1024
DEVELOPMENT_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
DEVELOPMENT_STORAGE_RESERVE_BYTES = 1024**3
COMPARATOR_SELECTION_RELATIVE_PATH = "artifacts/study/development/evaluation/comparator_selection.json"
COMPARATOR_SMOKE_RELATIVE_PATH = "artifacts/study/development/evaluation/comparator_smoke.json"
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ComparatorTuningError(RuntimeError):
    pass


ComparatorAdapterConfig: TypeAlias = (
    ALRAConfig | MAGICConfig | DCAConfig | SCVIConfig | SAVERConfig
    | SCZivaConfig | AFMFConfig | BiAEImputeConfig | SCCRConfig | SCSDaeConfig
)


_CONFIG_TYPES: Mapping[str, type[ComparatorAdapterConfig]] = MappingProxyType({
    "alra": ALRAConfig, "magic": MAGICConfig, "dca": DCAConfig,
    "scvi": SCVIConfig, "saver": SAVERConfig, "scziva": SCZivaConfig,
    "afmf": AFMFConfig, "biaeimpute": BiAEImputeConfig,
    "sccr": SCCRConfig, "scsdae": SCSDaeConfig,
})


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_payload(config: ComparatorAdapterConfig) -> dict[str, object]:
    value = asdict(config)
    if isinstance(config, DCAConfig):
        value["hidden_size"] = list(config.hidden_size)
    return value


def encode_comparator_configuration(
    config: ComparatorAdapterConfig,
) -> dict[str, object]:
    if type(config) not in set(_CONFIG_TYPES.values()):
        raise TypeError("config must be an exact comparator adapter dataclass")
    return _json_payload(config)


def decode_comparator_configuration(
    method_id: str,
    payload: Mapping[str, object],
    *,
    expected_payload_sha256: str,
) -> ComparatorAdapterConfig:
    """Bootstrap the authority loader; Task 2 replaces this with strict type checks."""
    config_type = _CONFIG_TYPES.get(method_id)
    if config_type is None or type(payload) not in {dict, MappingProxyType}:
        raise ComparatorTuningError("comparator method or payload is invalid")
    observed = dict(payload)
    defaults = _json_payload(config_type())
    if set(observed) != set(defaults):
        raise ComparatorTuningError("comparator payload differs from its complete field set")
    constructor = dict(observed)
    if method_id == "dca":
        hidden = observed["hidden_size"]
        if type(hidden) is not list:
            raise ComparatorTuningError("DCA hidden_size must be a JSON array")
        constructor["hidden_size"] = tuple(hidden)
    if canonical_sha256(observed) != expected_payload_sha256:
        raise ComparatorTuningError("comparator payload checksum differs")
    try:
        decoded = config_type(**constructor)
    except (TypeError, ValueError) as error:
        raise ComparatorTuningError("comparator payload violates its adapter contract") from error
    if encode_comparator_configuration(decoded) != observed:
        raise ComparatorTuningError("comparator payload changed during typed normalization")
    return decoded


@dataclass(frozen=True, slots=True)
class ComparatorConfiguration:
    method_id: str
    configuration_id: str
    payload_json: str
    payload_sha256: str
    is_upstream_default: bool

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.payload_json))

    @property
    def observed_payload_sha256(self) -> str:
        return canonical_sha256(dict(self.payload))

    @property
    def payload_for_json_comparison(self) -> dict[str, object]:
        return _json_payload(self.decode())

    def decode(self) -> ComparatorAdapterConfig:
        return decode_comparator_configuration(
            self.method_id, self.payload,
            expected_payload_sha256=self.payload_sha256,
        )


@dataclass(frozen=True, slots=True)
class ComparatorTuningAuthority:
    schema_version: int
    contract_id: str
    method_order: tuple[str, ...]
    configurations: tuple[ComparatorConfiguration, ...]
    scheduled_same_input_ids: tuple[str, ...]
    required_control_ids: tuple[str, ...]
    established_comparator_ids: tuple[str, ...]
    modern_core_ids: tuple[str, ...]
    model_seeds: tuple[int, ...]
    selection_metrics: tuple[str, ...]
    receipt_path: str
    smoke_receipt_path: str
    file_sha256: str
    payload_sha256: str

    def configurations_for(self, method_id: str) -> tuple[ComparatorConfiguration, ...]:
        return tuple(row for row in self.configurations if row.method_id == method_id)


def _require_exact_mapping(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ComparatorTuningError(f"{name} has missing or extra fields")
    return value


def parse_comparator_tuning_authority(
    payload: object,
    *,
    registry: MethodRegistry,
    file_sha256: str,
) -> ComparatorTuningAuthority:
    root = _require_exact_mapping(payload, {
        "schema_version", "contract_id", "scope", "method_order",
        "scheduled_same_input_ids", "required_control_ids",
        "established_comparator_ids", "modern_core_ids", "model_seeds",
        "configurations", "selection", "budgets", "storage", "smoke",
        "payload_sha256",
    }, "comparator tuning authority")
    unsigned = {key: value for key, value in root.items() if key != "payload_sha256"}
    if root["schema_version"] != 1 or root["contract_id"] != "maskimpute-comparator-tuning-v1":
        raise ComparatorTuningError("comparator tuning schema or contract differs")
    if root["payload_sha256"] != canonical_sha256(unsigned):
        raise ComparatorTuningError("comparator tuning payload checksum differs")
    method_order = tuple(root["method_order"]) if type(root["method_order"]) is list else ()
    expected_order = tuple(_CONFIG_TYPES)
    if method_order != expected_order:
        raise ComparatorTuningError("comparator tuning method order differs")
    rows = root["configurations"]
    if type(rows) is not list:
        raise ComparatorTuningError("comparator configurations must be an array")
    configurations: list[ComparatorConfiguration] = []
    for index, raw in enumerate(rows):
        row = _require_exact_mapping(raw, {
            "method_id", "configuration_id", "is_upstream_default", "payload", "payload_sha256",
        }, f"comparator configuration {index}")
        if row["method_id"] not in _CONFIG_TYPES or not isinstance(row["configuration_id"], str):
            raise ComparatorTuningError("comparator configuration identity is invalid")
        if _SAFE_ID.fullmatch(row["configuration_id"]) is None or type(row["is_upstream_default"]) is not bool:
            raise ComparatorTuningError("comparator configuration identity is invalid")
        decoded = decode_comparator_configuration(
            row["method_id"], row["payload"],
            expected_payload_sha256=row["payload_sha256"],
        )
        configurations.append(ComparatorConfiguration(
            method_id=row["method_id"],
            configuration_id=row["configuration_id"],
            payload_json=_canonical_bytes(_json_payload(decoded)).decode("utf-8"),
            payload_sha256=row["payload_sha256"],
            is_upstream_default=row["is_upstream_default"],
        ))
    scheduled = tuple(
        spec.id for spec in registry.methods
        if spec.execution_scope == "same_input_required" and spec.role != "candidate"
    )
    if tuple(root["scheduled_same_input_ids"]) != scheduled:
        raise ComparatorTuningError("scheduled same-input denominator differs from registry")
    selection = _require_exact_mapping(root["selection"], {
        "metrics", "collapse_order", "prezero_mechanism", "pareto_rule",
        "rank_rule", "selection_tuple", "readiness", "receipt_path",
    }, "comparator selection policy")
    smoke = _require_exact_mapping(root["smoke"], {
        "receipt_path", "cells", "genes", "model_seed", "batch_rule",
        "count_formula", "projection_multiplier", "output_retention",
    }, "comparator smoke policy")
    storage = _require_exact_mapping(root["storage"], {
        "max_log_receipt_bytes", "max_executor_receipt_bytes", "max_record_bytes",
        "max_checkpoint_bytes", "reserve_bytes",
    }, "development storage policy")
    if storage != {
        "max_log_receipt_bytes": DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
        "max_executor_receipt_bytes": DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
        "max_record_bytes": DEVELOPMENT_MAX_RECORD_BYTES,
        "max_checkpoint_bytes": DEVELOPMENT_MAX_CHECKPOINT_BYTES,
        "reserve_bytes": DEVELOPMENT_STORAGE_RESERVE_BYTES,
    }:
        raise ComparatorTuningError("development storage policy differs")
    return ComparatorTuningAuthority(
        schema_version=1,
        contract_id=root["contract_id"],
        method_order=method_order,
        configurations=tuple(configurations),
        scheduled_same_input_ids=tuple(root["scheduled_same_input_ids"]),
        required_control_ids=tuple(root["required_control_ids"]),
        established_comparator_ids=tuple(root["established_comparator_ids"]),
        modern_core_ids=tuple(root["modern_core_ids"]),
        model_seeds=tuple(root["model_seeds"]),
        selection_metrics=tuple(selection["metrics"]),
        receipt_path=selection["receipt_path"],
        smoke_receipt_path=smoke["receipt_path"],
        file_sha256=file_sha256,
        payload_sha256=root["payload_sha256"],
    )
```

The displayed parser is the structural spine, not permission to leave nested
blocks open. In this same step, add closed validators and mutation tests for
every field below. The loader must reject a changed value even when an attacker
recomputes both the row hashes and the authority payload hash:

```python
EXPECTED_SCOPE = {"data_scope": "development_only", "final_data_used": False}
EXPECTED_MODEL_SEEDS = (42, 43, 44)
EXPECTED_METRICS = (
    "mse", "mse_dropout", "gnrmse", "mse_pre_dropout_zero",
    "corr_err", "mse_non_dropout_nonzero",
)
EXPECTED_COLLAPSE_ORDER = (
    "mean_model_seeds_within_dataset_view",
    "mean_paired_views_within_biological_draw",
    "retain_biological_draw_units",
)
EXPECTED_SELECTION_TUPLE = (
    "maximum_metric_rank_quarters", "sum_metric_rank_quarters",
    "mse_rank_quarters", "mse_dropout_rank_quarters",
    "gnrmse_rank_quarters", "mse_pre_dropout_zero_rank_quarters",
    "corr_err_rank_quarters", "mse_non_dropout_nonzero_rank_quarters",
    "upstream_default_penalty", "configuration_id",
)
EXPECTED_BUDGETS = {
    "max_configurations_per_method": 20,
    "gpu_seconds_per_method": 28_800,
    "cpu_seconds_per_method": 86_400,
    "per_run_timeout_seconds": 21_600,
    "max_rss_bytes": 48 * 1024**3,
    "max_gpu_bytes": 14 * 1024**3,
    "intrinsic_terminal_statuses": [
        "failed", "timeout", "resource_exceeded", "unavailable",
    ],
    "blocking_statuses": [
        "budget_exhausted", "blocked_authority", "infrastructure_error",
    ],
}
EXPECTED_SMOKE = {
    "receipt_path": COMPARATOR_SMOKE_RELATIVE_PATH,
    "cells": 900,
    "genes": 500,
    "model_seed": 42,
    "batch_rule": "alternating_batch-0_batch-1",
    "count_formula": "(17*cell+31*gene+7*(cell^gene))%6",
    "projection_multiplier": 48,
    "output_retention": "discarded_without_evaluator_or_metrics",
}
```

Require the exact 34 `(method_id, configuration_id)` pairs in the normative
order, no duplicate configuration ID or payload identity within a method, one
and only one `is_upstream_default=True` row per method, and that row's encoded
payload equal `encode_comparator_configuration(Config())`. Require the four
global sets shown in Step 2, `prezero_mechanism="symsim"`, the exact Pareto and
rank-rule identifiers, readiness thresholds of both controls, all five
established methods, and `minimum_modern_core_selectable=3`. Validate `scope`,
`selection`, `budgets`, `storage`, and `smoke` by exact key set and exact value;
never merely deserialize them.

- [ ] **Step 5: Materialize the exact tracked JSON**

Create `study/comparator_tuning.json` as canonical two-space JSON with the exact top-level fields parsed above. Use the four global comparator sets, metrics, budgets, storage constants, smoke definition, and configuration IDs from this plan. The normative 34-row payload/hash table is printed in Task 2 Step 4 for readability, but it is input to this step: materialize all of those complete payloads now, before running Task 1 tests. Compute `payload_sha256` over the object with that field omitted, then insert it as the final field. Verify the self-hash without modifying the file:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /tmp/maskimpute-supported/bin/python - <<'PY'
import json
from pathlib import Path
from maskimpute_benchmark.protocol import canonical_sha256

path = Path("study/comparator_tuning.json")
value = json.loads(path.read_text())
observed = value.pop("payload_sha256")
assert observed == canonical_sha256(value)
assert path.read_bytes() == json.dumps({**value, "payload_sha256": observed}, indent=2).encode() + b"\n"
print(observed)
PY
```

Expected: one 64-character lowercase hexadecimal digest and exit status 0.

- [ ] **Step 6: Finish strict repository loading**

Add this loader to `comparator_tuning.py`:

```python
def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparatorTuningError(f"duplicate comparator authority key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ComparatorTuningError(f"nonfinite comparator authority constant {value}")


def load_comparator_tuning_authority(
    repository: Path,
    *,
    registry: MethodRegistry,
    require_clean: bool = True,
) -> ComparatorTuningAuthority:
    if not isinstance(repository, Path) or not isinstance(registry, MethodRegistry):
        raise TypeError("repository and registry have invalid types")
    root = repository.resolve(strict=True)
    path = root / "study/comparator_tuning.json"
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ComparatorTuningError("comparator tuning authority is invalid") from error
    if raw != json.dumps(payload, indent=2).encode("utf-8") + b"\n":
        raise ComparatorTuningError("comparator tuning authority is not canonical tracked JSON")
    if require_clean:
        import subprocess
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "study/comparator_tuning.json"],
            cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--",
             "study/comparator_tuning.json"],
            cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if tracked.returncode != 0 or status.returncode != 0 or status.stdout:
            raise ComparatorTuningError("comparator tuning authority is not tracked and clean")
    return parse_comparator_tuning_authority(
        payload, registry=registry, file_sha256=hashlib.sha256(raw).hexdigest(),
    )
```

- [ ] **Step 7: Run focused authority tests**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py::test_tracked_authority_has_exact_grid_and_operational_contract \
  -q -W error -p no:cacheprovider
```

Before the Task 1 commit, make the test call
`load_comparator_tuning_authority(ROOT, registry=registry, require_clean=False)`;
expected result is PASS. After the commit, add a separate
`test_clean_repository_loads_comparator_tuning_authority` using the default
`require_clean=True` and run that node before starting Task 2.

- [ ] **Step 8: Commit the authority contract**

```bash
git add docs/superpowers/specs/2026-07-18-fair-comparator-tuning-design.md \
  study/comparator_tuning.json maskimpute_benchmark/comparator_tuning.py \
  tests/test_comparator_tuning.py
git commit -m "feat: add comparator tuning authority"
```

---

### Task 2: Decode all 34 complete payloads into exact adapter dataclasses

**Files:**

- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Modify: `tests/test_comparator_tuning.py`

**Interfaces:**

- Consumes: `_CONFIG_TYPES`, `_json_payload`, and each `ComparatorConfiguration.payload_sha256` from Task 1.
- Produces: `decode_comparator_configuration(method_id, payload, *, expected_payload_sha256) -> ComparatorAdapterConfig` and `encode_comparator_configuration(config) -> dict[str, object]`.

- [ ] **Step 1: Add the failing table-driven decoder test**

```python
import copy
import pytest

from maskimpute_benchmark.comparator_tuning import (
    ComparatorTuningError,
    decode_comparator_configuration,
    encode_comparator_configuration,
)


def test_decode_comparator_configuration_is_closed_and_exact() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(ROOT, registry=registry)
    for row in authority.configurations:
        decoded = row.decode()
        assert encode_comparator_configuration(decoded) == dict(row.payload)

        missing = dict(row.payload)
        missing.pop(next(iter(missing)))
        with pytest.raises(ComparatorTuningError, match="complete field set"):
            decode_comparator_configuration(
                row.method_id, missing,
                expected_payload_sha256=row.payload_sha256,
            )

        extra = {**dict(row.payload), "unexpected": 1}
        with pytest.raises(ComparatorTuningError, match="complete field set"):
            decode_comparator_configuration(
                row.method_id, extra,
                expected_payload_sha256=row.payload_sha256,
            )

    magic = authority.configurations_for("magic")[0]
    bool_as_int = {**dict(magic.payload), "knn": True}
    with pytest.raises(ComparatorTuningError, match="primitive type"):
        decode_comparator_configuration(
            "magic", bool_as_int,
            expected_payload_sha256=magic.payload_sha256,
        )

    dca = authority.configurations_for("dca")[0]
    tuple_payload = {**dict(dca.payload), "hidden_size": (64, 32, 64)}
    with pytest.raises(ComparatorTuningError, match="JSON array"):
        decode_comparator_configuration(
            "dca", tuple_payload,
            expected_payload_sha256=dca.payload_sha256,
        )
```

- [ ] **Step 2: Run the decoder test red**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py::test_decode_comparator_configuration_is_closed_and_exact \
  -q -W error -p no:cacheprovider
```

Expected: FAIL on at least the bool-as-int adversarial case because Task 1's bootstrap decoder establishes the authority but does not yet enforce exact JSON primitive types.

- [ ] **Step 3: Replace the bootstrap codec with exact primitive-type validation and round-trip hashing**

```python
def _primitive_type_matches(observed: object, default: object) -> bool:
    if default is None:
        return observed is None
    if type(default) is float:
        return type(observed) is float and math.isfinite(observed)
    return type(observed) is type(default)


def encode_comparator_configuration(
    config: ComparatorAdapterConfig,
) -> dict[str, object]:
    if type(config) not in set(_CONFIG_TYPES.values()):
        raise TypeError("config must be an exact comparator adapter dataclass")
    return _json_payload(config)


def decode_comparator_configuration(
    method_id: str,
    payload: Mapping[str, object],
    *,
    expected_payload_sha256: str,
) -> ComparatorAdapterConfig:
    config_type = _CONFIG_TYPES.get(method_id)
    if config_type is None or type(payload) not in {dict, MappingProxyType}:
        raise ComparatorTuningError("comparator method or payload is invalid")
    defaults = _json_payload(config_type())
    observed = dict(payload)
    if set(observed) != set(defaults):
        raise ComparatorTuningError("comparator payload differs from its complete field set")
    constructor = dict(observed)
    for name, default in defaults.items():
        value = observed[name]
        if method_id == "dca" and name == "hidden_size":
            if (
                type(value) is not list
                or not value
                or any(type(item) is not int or item <= 0 for item in value)
            ):
                raise ComparatorTuningError("DCA hidden_size must be a positive-integer JSON array")
            constructor[name] = tuple(value)
            continue
        if not _primitive_type_matches(value, default):
            raise ComparatorTuningError(f"comparator field {name} has the wrong primitive type")
    if canonical_sha256(observed) != expected_payload_sha256:
        raise ComparatorTuningError("comparator payload checksum differs")
    try:
        decoded = config_type(**constructor)
    except (TypeError, ValueError) as error:
        raise ComparatorTuningError("comparator payload violates its adapter contract") from error
    encoded = encode_comparator_configuration(decoded)
    if encoded != observed or canonical_sha256(encoded) != expected_payload_sha256:
        raise ComparatorTuningError("comparator payload changed during typed normalization")
    return decoded
```

- [ ] **Step 4: Revalidate the normative full payload table already materialized in Task 1**

Task 1 Step 5 uses this normative table when it creates the authority. Recompute it from the strict Task 2 codec, require byte-for-byte equality with the tracked payloads, and vary only the named axis. The row hashes are canonical JSON SHA-256:

```text
alra-default 19610b1cad02cd7ce2c5e1ad0e6e3764b78056a52ce45e58d64fb089620676b6
  {"k":0,"q":10,"quantile_probability":0.001,"use_mkl":false}

magic-t03 a887711d7c3972023d80d368fc299b78827edf4ebb1d7aa34680d7bb62c36a1a
magic-t01 68a815706525462b1c7e0fa81ca405240121c3dbd78f9cb1c239ea97892f9ecf
magic-t05 803a2e42fda0686c6580b1263953dfd086d76c21bf863691e9e07d0a24f0203e
magic-t07 2ca13dad3342a2e596dda17cb3b2d989f09d1b21e04bdde7c5d72dbb2343b01c
  {"knn":5,"knn_max":null,"decay":1,"diffusion_time":3|1|5|7,"n_pca":100,"solver":"exact","distance":"euclidean","n_jobs":1}

dca-h64-32-64 0bec7e04f7a8dd7a962ff7a8aaf3e1a439414b4cf831e3e169b21856cf2b6dee
dca-h32-16-32 03a4787dafe21f5e941d6bfeba6139a9cba828295cb7d181b69796a03d0aa339
dca-h32-32 d5d178f2cc2f3eabe210b60c6c227b91fca4bc6d4e576f5a10349b153de3d28e
dca-h64-64 838530f73bd7a9e7e8cceac208570973791f713cbed791ff7dc576af3c91b396
  {"ae_type":"zinb-conddisp","normalize_per_cell":true,"scale":true,"log1p":true,"hidden_size":[64,32,64]|[32,16,32]|[32,32]|[64,64],"hidden_dropout":0.0,"batchnorm":true,"activation":"relu","initializer":"glorot_uniform","epochs":300,"reduce_lr":10,"early_stop":15,"batch_size":32,"optimizer":"RMSprop"}

scvi-z10 53357dea2c15452bdc1116f8ac6c567a6e22846c630756b7518cf389ab4b6dc0
scvi-z05 838419aae886e92c06be4f139ebe3899a0b23e0de2ccecadcac7256b64a77549
scvi-z20 309a22e8553599ebbf4e10aca6796597374dcf655418ecf046378cb271847caa
scvi-z30 9107209f7b74161deeb64f4dca4b877404c8d7a0c58e188022f93622a64bfb1e
  {"n_hidden":128,"n_latent":10|5|20|30,"n_layers":1,"dropout_rate":0.1,"dispersion":"gene","gene_likelihood":"zinb","use_observed_lib_size":true,"latent_distribution":"normal","max_epochs":null,"batch_size":128,"batch_key":"batch"}

saver-default 9bd72a724b3d8b4ab52bd7aa17b7d22d15cc90ca2b52fe5856a8b456377695fc
  {"do_fast":true,"ncores":1,"size_factor":null,"estimates_only":true}

scziva-tau-0p001 57e8b1a6de708b01c372df126c24c12dd8114cdc4d4295063207470632804c9c
scziva-tau-0p0001 2cc430a3b0265bd19a0acf18f3b020f6160570844b6adcb8c54807694aa9c2c6
scziva-tau-0p01 a500507cac9282b76c8389bdbbf1bbea5459303742f0d03dd1183f73e9eec74c
scziva-tau-0p05 accdefa2aa562cf9eb78706e7d5ace0903bad14ef132f76361e897326d0110bb
  {"num_epochs":200,"learning_rate":0.001,"hidden_dim":128,"latent_dim":64,"use_cnn":true,"tau":0.001|0.0001|0.01|0.05,"auxiliary_weight_min":0.5,"auxiliary_weight_max":1.5,"auxiliary_regularization":0.001,"reorder_genes":true,"device":null}

afmf-sigma-3 4cb595af856d55eb043d9df1c7e930999fe9892f2d9ccaf7c63f280afdde0809
afmf-sigma-1 c6ee030b9037603ed604471b99b635fabde93d63afe3a334cb3baab43d17aa46
afmf-sigma-2 0a27ef3b006f454ad13b3d4fb453ad7d4e4741cbc95a09027db414ca014705de
afmf-sigma-4 f5672f1b8718e2c792bc1137274bdf7972c229e3f0bb01240df5bf4a067710fb
  {"iterations":10000,"tolerance":0.0001,"lambda_p":0.0,"lambda_q":0.0,"sigma":3.0|1.0|2.0|4.0}

biaeimpute-z128 813d0d6713c790edc2586ed8ea32d79e04b7b4ce6e496c7ea86726b1a655d9b3
biaeimpute-z32 d14fd645eb952f9f3ddfb88de626c9aacbc685a907d453c1b194193b8bec557b
biaeimpute-z64 87bc0ecf181a8b887f595dbe1ebac6277067c297e6c91bdb4f06db888f1589ef
biaeimpute-z256 0a7191a92c8591e2690b1b12855a13c8d314c05e7a993f48e81c41c85922c3c1
  {"epochs":500,"latent_size":128|32|64|256,"learning_rate":0.0002,"beta1":0.9,"beta2":0.999,"row_batch_size":31,"column_batch_size":200,"mask_ratio":0.0,"device":null}

sccr-k15 767f9c13feb0d7583af5b7253fed62e7ff83e538e99f441992ec2cfea7db7ee9
sccr-k05 e06f30b5673c7fec19083a4e89df73157c2aa0ad5c8d6f6fec17aadb9104778a
sccr-k10 65aca3a3c0f3388f44da8eae0bdbc537e2fc17e8d7e292692faab178b5fc8667
sccr-k30 4cef1fb8d3e85e1ddbcc3924c0a2ad6c72dd66b1542716ee1e071080f5dfca42
  {"neighbors":15|5|10|30,"gene_neighbors":2,"symmetric_final_graph":true,"iterations":40,"complete_relation_weight":0.05,"soft_propagation_weight":0.99,"final_blend_weight":0.01,"device":null}

scsdae-zero-1 414b77eead7f8f01c8379039e9b6df676388f1d921dfbdcd65eb804f79639ae0
scsdae-zero-0p25 09ee7b4cc3df0ecc8f90fb146e2c640fcf75f772d491a3c8f132847a2ae8eb5f
scsdae-zero-0p5 c0f3865e08c710700900a8cd6fb49b32f7f1e8cb3ec637b2f2a746c9e1392d0e
scsdae-zero-0p75 cfdd9ab34fd1d14f0251714d10aa9fdafdceeb0edf5a2820bb32c7cbcf2539c6
  {"batch_size":256,"autoencoder_iterations":2000,"pretrain_iterations":1000,"zero_loss_weight":1.0|0.25|0.5|0.75,"observed_loss_weight":1.0,"dropout_rate":0.2,"l1_regularization":0.0,"l2_regularization":0.0,"gene_scale":false,"gpu_index":0}
```

The vertical bars above enumerate separate complete row payloads; no tracked JSON value contains a vertical bar.

- [ ] **Step 5: Run all authority/decoder tests green**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'authority or decode' \
  -q -W error -p no:cacheprovider
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit typed decoding**

```bash
git add maskimpute_benchmark/comparator_tuning.py study/comparator_tuning.json \
  tests/test_comparator_tuning.py
git commit -m "feat: decode closed comparator configurations"
```

---

### Task 3: Bind registry-derived comparator sets into selection authority

**Files:**

- Modify: `study/selection_contract.json`
- Modify: `study/development_search.json`
- Modify: `maskimpute_benchmark/selection.py:76-84,251-278,1988-2201,2375-2412,2630-2710`
- Modify: `tests/test_selection_authority.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_method_registry.py`

**Interfaces:**

- Consumes: `load_comparator_tuning_authority(...)` from Task 1 and the raw/payload SHA-256 of `study/comparator_tuning.json`.
- Produces: `SelectionAuthority.scheduled_same_input_ids`, `.required_control_ids`, `.established_comparator_ids`, `.modern_core_ids`, `.comparator_tuning_file_sha256`, and `.comparator_tuning_payload_sha256`; removes `required_comparator_ids` as an authority field.

- [ ] **Step 1: Write failing selection-authority regressions**

```python
def test_selection_authority_uses_exact_comparator_readiness_sets() -> None:
    authority = _load_selection_authority(ROOT, require_clean=False)
    assert authority.scheduled_same_input_ids == (
        "observed", "capacity-matched-ae", "alra", "magic", "dca", "scvi",
        "saver", "scziva", "afmf", "biaeimpute", "sccr", "scsdae",
    )
    assert authority.required_control_ids == ("observed", "capacity-matched-ae")
    assert authority.established_comparator_ids == ("alra", "magic", "dca", "scvi", "saver")
    assert authority.modern_core_ids == ("scziva", "afmf", "biaeimpute", "sccr")
    assert "biaeimpute" in authority.scheduled_same_input_ids
    assert len(authority.comparator_tuning_file_sha256) == 64
    assert len(authority.comparator_tuning_payload_sha256) == 64


def test_selection_authority_rejects_biaeimpute_omission(tmp_path: Path) -> None:
    repository = _ready_repository(tmp_path)
    contract_path = repository / "study/selection_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["scheduled_same_input_ids"].remove("biaeimpute")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    with pytest.raises(SelectionAuthorityError, match="scheduled same-input denominator"):
        _load_selection_authority(repository, require_clean=False)
```

- [ ] **Step 2: Run the tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_selection_authority.py \
  -k 'comparator_readiness_sets or biaeimpute_omission' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because `SelectionAuthority` still exposes only `required_comparator_ids`.

- [ ] **Step 3: Replace the selection-contract comparator fields**

Replace `required_comparator_ids` in `study/selection_contract.json` with:

```json
"comparator_tuning_path": "study/comparator_tuning.json",
"scheduled_same_input_ids": ["observed", "capacity-matched-ae", "alra", "magic", "dca", "scvi", "saver", "scziva", "afmf", "biaeimpute", "sccr", "scsdae"],
"required_control_ids": ["observed", "capacity-matched-ae"],
"established_comparator_ids": ["alra", "magic", "dca", "scvi", "saver"],
"modern_core_ids": ["scziva", "afmf", "biaeimpute", "sccr"]
```

Compute the two omitted values with the read-only block below. Use `apply_patch`
to add `comparator_tuning_file_sha256` with `raw_file_sha256` and
`comparator_tuning_payload_sha256` with `payload_sha256`; never write a
descriptive sentinel into tracked JSON. Then update
`study/development_search.json.authority` with those exact values and the new
raw `selection_contract_sha256`. Verify all child bindings with:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /tmp/maskimpute-supported/bin/python - <<'PY'
import hashlib, json
from pathlib import Path

tuning_path = Path("study/comparator_tuning.json")
contract_path = Path("study/selection_contract.json")
search = json.loads(Path("study/development_search.json").read_text())
contract = json.loads(contract_path.read_text())
tuning = json.loads(tuning_path.read_text())
raw_file_sha256 = hashlib.sha256(tuning_path.read_bytes()).hexdigest()
payload_sha256 = tuning["payload_sha256"]
print({"raw_file_sha256": raw_file_sha256, "payload_sha256": payload_sha256})
assert contract["comparator_tuning_file_sha256"] == hashlib.sha256(tuning_path.read_bytes()).hexdigest()
assert contract["comparator_tuning_payload_sha256"] == tuning["payload_sha256"]
assert search["authority"]["selection_contract_sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
assert search["authority"]["comparator_tuning_file_sha256"] == contract["comparator_tuning_file_sha256"]
assert search["authority"]["comparator_tuning_payload_sha256"] == tuning["payload_sha256"]
PY
```

Expected: exit status 0 and no output.

- [ ] **Step 4: Update `SelectionAuthority` and closed loading**

Use these exact dataclass fields in `selection.py`:

```python
@dataclass(frozen=True, slots=True)
class SelectionAuthority:
    mechanisms: tuple[str, ...]
    biological_ids: tuple[str, ...]
    technical_views: tuple[str, ...]
    model_seeds: tuple[int, ...]
    scheduled_same_input_ids: tuple[str, ...]
    required_control_ids: tuple[str, ...]
    established_comparator_ids: tuple[str, ...]
    modern_core_ids: tuple[str, ...]
    comparator_tuning_file_sha256: str
    comparator_tuning_payload_sha256: str
    attempts: tuple[CandidateAttempt, ...]
    declarations: tuple[MethodDeclaration, ...]
    endpoint_policies: tuple[EndpointPolicy, ...]
    revision_policy: RevisionPolicy
    exclusions: tuple[SearchExclusion, ...]
    method_bindings: Mapping[str, str]
    base_maskimpute_config: Mapping[str, Any]
    base_maskimpute_config_sha256: str
    count_model_config: Mapping[str, Any]
    count_model_config_sha256: str
    dataset_qc_policy: Mapping[str, Any]
    dataset_qc_policy_sha256: str
    ablation_specs: tuple[Mapping[str, Any], ...]
    ablation_spec_ids: tuple[str, ...]
    ablation_run_keys: tuple[tuple[str, int], ...]
    calibration_equivalence_reason: str | None
    calibration_effect_status: str
    retained_calibration: RetainedCalibrationBinding
    count_score_manifest: RetainedCalibrationBinding
    file_sha256: Mapping[str, str]
```

Add `study/comparator_tuning.json` to `_AUTHORITY_PATHS`, load it with `load_comparator_tuning_authority`, require the four contract arrays to equal the tuning authority arrays, require the raw and payload hashes to match, and derive noncandidate declarations with `required_for_claim=False`. Candidate gates will set the exact ready population after receipt validation in Task 10; static declarations must not predeclare unavailable comparator claims.

Extend the existing `_ready_repository(tmp_path)` test helper to copy
`study/comparator_tuning.json` and recompute its selection-contract and
development-search descendant hashes. Precommit repository-copy tests use
`require_clean=False`; the production-checkout clean-loader regression runs
only after this task's commit.

Do not break the 42 existing downstream consumers before Tasks 12--16 migrate
their semantics. Retain this temporary read-only compatibility property and
forbid new code from using it:

```python
    @property
    def required_comparator_ids(self) -> tuple[str, ...]:
        return tuple(
            method_id for method_id in self.scheduled_same_input_ids
            if method_id != "biaeimpute"
        )
```

Task 16 removes the property after the final consumer has moved to
`scheduled_same_input_ids` and `ready_comparison_population_ids`.

- [ ] **Step 5: Run adjacent selection and registry tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_selection_authority.py tests/test_method_registry.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS, including the exact BiAEImpute denominator regression.

- [ ] **Step 6: Commit authority descendants**

```bash
git add study/selection_contract.json study/development_search.json \
  maskimpute_benchmark/selection.py tests/test_selection_authority.py \
  tests/test_method_registry.py
git commit -m "feat: bind comparator readiness authority"
```

---

### Task 4: Derive collision-resistant preselection comparator identities

**Files:**

- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Modify: `maskimpute_benchmark/runner.py:635-747,1424-1447,2126-2405`
- Modify: `tests/test_comparator_tuning.py`
- Modify: `tests/test_benchmark_runner.py`

**Interfaces:**

- Consumes: `ComparatorConfiguration`, `ComparatorTuningAuthority`, `MethodSpec`, the development runtime-lock file hash, and `ExecutionEnvironmentRegistry.registry_sha256`.
- Produces: `BoundComparatorConfiguration`, `bind_comparator_configuration_identity(...)`, comparator identity fields on `AuthorizedConfiguration`, `RunPlanEntry`, and `ExecutionRequest`.

- [ ] **Step 1: Write a failing mutation test for every stable identity field**

```python
from dataclasses import replace

from maskimpute_benchmark.comparator_tuning import (
    bind_comparator_configuration_identity,
)


def test_configuration_method_identity_binds_every_stable_field() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(ROOT, registry=registry)
    configuration = authority.configurations_for("magic")[0]
    spec = registry.by_id("magic")
    bound = bind_comparator_configuration_identity(
        configuration,
        spec,
        authority,
        runtime_lock_sha256="1" * 64,
        environment_registry_sha256="2" * 64,
    )
    stable_fields = (
        "registry_method_sha256",
        "configuration_payload_sha256",
        "tuning_authority_file_sha256",
        "tuning_authority_payload_sha256",
        "source_authority_sha256",
        "runtime_lock_sha256",
        "environment_registry_sha256",
    )
    observed = set()
    for field in stable_fields:
        mutated = replace(bound, **{field: "f" * 64})
        observed.add(mutated.recomputed_identity_sha256)
        assert mutated.recomputed_identity_sha256 != bound.configuration_method_identity_sha256
    assert len(observed) == len(stable_fields)


def test_configuration_method_identities_do_not_collide_within_method() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(ROOT, registry=registry)
    identities = [
        bind_comparator_configuration_identity(
            row, registry.by_id(row.method_id), authority,
            runtime_lock_sha256="1" * 64,
            environment_registry_sha256="2" * 64,
        ).configuration_method_identity_sha256
        for row in authority.configurations_for("magic")
    ]
    assert len(identities) == len(set(identities)) == 4
```

- [ ] **Step 2: Run the identity tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'configuration_method_identity' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because `BoundComparatorConfiguration` and the binding function are absent.

- [ ] **Step 3: Implement the exact domain-separated identity**

Add to `comparator_tuning.py`:

```python
from dataclasses import asdict
from .methods.base import MethodSpec


@dataclass(frozen=True, slots=True)
class BoundComparatorConfiguration:
    configuration: ComparatorConfiguration
    registry_method_sha256: str
    configuration_payload_sha256: str
    tuning_authority_file_sha256: str
    tuning_authority_payload_sha256: str
    source_authority_sha256: str
    runtime_lock_sha256: str
    environment_registry_sha256: str
    configuration_method_identity_sha256: str

    @property
    def identity_body(self) -> dict[str, object]:
        return {
            "schema": "maskimpute-comparator-configuration-method-identity-v1",
            "registry_method_sha256": self.registry_method_sha256,
            "configuration_payload_sha256": self.configuration_payload_sha256,
            "tuning_authority_file_sha256": self.tuning_authority_file_sha256,
            "tuning_authority_payload_sha256": self.tuning_authority_payload_sha256,
            "source_authority_sha256": self.source_authority_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "environment_registry_sha256": self.environment_registry_sha256,
        }

    @property
    def recomputed_identity_sha256(self) -> str:
        return canonical_sha256(self.identity_body)


def bind_comparator_configuration_identity(
    configuration: ComparatorConfiguration,
    method_spec: MethodSpec,
    authority: ComparatorTuningAuthority,
    *,
    runtime_lock_sha256: str,
    environment_registry_sha256: str,
) -> BoundComparatorConfiguration:
    if configuration.method_id != method_spec.id:
        raise ComparatorTuningError("configuration method differs from registry method")
    registry_method_sha256 = canonical_sha256(asdict(method_spec))
    source_authority_sha256 = canonical_sha256({
        "schema": "maskimpute-comparator-source-authority-v1",
        "method_id": method_spec.id,
        "source": asdict(method_spec.source),
    })
    body = {
        "schema": "maskimpute-comparator-configuration-method-identity-v1",
        "registry_method_sha256": registry_method_sha256,
        "configuration_payload_sha256": configuration.payload_sha256,
        "tuning_authority_file_sha256": authority.file_sha256,
        "tuning_authority_payload_sha256": authority.payload_sha256,
        "source_authority_sha256": source_authority_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "environment_registry_sha256": environment_registry_sha256,
    }
    return BoundComparatorConfiguration(
        configuration=configuration,
        registry_method_sha256=registry_method_sha256,
        configuration_payload_sha256=configuration.payload_sha256,
        tuning_authority_file_sha256=authority.file_sha256,
        tuning_authority_payload_sha256=authority.payload_sha256,
        source_authority_sha256=source_authority_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        environment_registry_sha256=environment_registry_sha256,
        configuration_method_identity_sha256=canonical_sha256(body),
    )
```

- [ ] **Step 4: Extend `AuthorizedConfiguration` without weakening legacy kinds**

Add nullable identity fields with defaults to `AuthorizedConfiguration`, add `comparator_tuning` and `comparator_nonexecution` to its closed kind set, and add this constructor:

```python
    registry_method_sha256: str | None = None
    tuning_authority_file_sha256: str | None = None
    tuning_authority_payload_sha256: str | None = None
    source_authority_sha256: str | None = None
    runtime_lock_sha256: str | None = None
    environment_registry_sha256: str | None = None
    configuration_method_identity_sha256: str | None = None
    nonexecution_identity_sha256: str | None = None

    @classmethod
    def from_bound_comparator(
        cls, bound: BoundComparatorConfiguration
    ) -> AuthorizedConfiguration:
        row = bound.configuration
        return cls.create(
            method_id=row.method_id,
            configuration_id=row.configuration_id,
            kind="comparator_tuning",
            payload=dict(row.payload),
            requires_count_score=False,
            requires_calibration=False,
            configuration_sha256=row.payload_sha256,
            registry_method_sha256=bound.registry_method_sha256,
            tuning_authority_file_sha256=bound.tuning_authority_file_sha256,
            tuning_authority_payload_sha256=bound.tuning_authority_payload_sha256,
            source_authority_sha256=bound.source_authority_sha256,
            runtime_lock_sha256=bound.runtime_lock_sha256,
            environment_registry_sha256=bound.environment_registry_sha256,
            configuration_method_identity_sha256=(
                bound.configuration_method_identity_sha256
            ),
        )
```

Extend `create(...)` to accept those eight keyword fields and pass them into `cls`. In `__post_init__`, require all seven stable hashes plus `configuration_method_identity_sha256` exactly when `kind == "comparator_tuning"`, recompute the identity body above, and forbid all identity/nonexecution fields on `registry`, `candidate_search`, and `ablation`. Require only `nonexecution_identity_sha256` for `comparator_nonexecution`; it must never be executable.

Add these exact fields to `to_dict()` and to `RunPlanEntry`:

```python
    configuration_payload_sha256: str | None = None
    configuration_method_identity_sha256: str | None = None
    nonexecution_identity_sha256: str | None = None
```

For existing kinds set `configuration_payload_sha256` equal to `configuration_sha256`; for comparator tuning additionally require the method identity; for comparator nonexecution require a null method identity and a nonnull nonexecution identity.

- [ ] **Step 5: Bind the same fields into `ExecutionRequest` integrity**

Add the three fields above plus the six component hashes to `ExecutionRequest`, include them in `ExecutionRequest.create()`'s `values` mapping, and repeat them in `validate_integrity()`. Add this condition before dispatch:

```python
        if self.configuration_kind == "comparator_tuning":
            body = {
                "schema": "maskimpute-comparator-configuration-method-identity-v1",
                "registry_method_sha256": self.registry_method_sha256,
                "configuration_payload_sha256": self.configuration_payload_sha256,
                "tuning_authority_file_sha256": self.tuning_authority_file_sha256,
                "tuning_authority_payload_sha256": self.tuning_authority_payload_sha256,
                "source_authority_sha256": self.source_authority_sha256,
                "runtime_lock_sha256": self.runtime_lock_sha256,
                "environment_registry_sha256": self.environment_registry_sha256,
            }
            if self.configuration_method_identity_sha256 != canonical_sha256(body):
                raise RunnerContractError("comparator configuration identity mismatch")
```

- [ ] **Step 6: Run identity and request integrity tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'configuration_method_identity' \
  tests/test_benchmark_runner.py -k 'execution_request and identity' \
  -q -W error -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit identity binding**

```bash
git add maskimpute_benchmark/comparator_tuning.py maskimpute_benchmark/runner.py \
  tests/test_comparator_tuning.py tests/test_benchmark_runner.py
git commit -m "feat: bind comparator configuration identities"
```

---

### Task 5: Expand and order the exact 2,896-row development plan

**Files:**

- Modify: `maskimpute_benchmark/runner.py:883-1110,1424-1624,6720-6758`
- Modify: `tests/test_benchmark_runner.py:390-510`
- Modify: `tests/test_runtime_environments.py`

**Interfaces:**

- Consumes: selection authority comparator hashes, `ComparatorTuningAuthority`, `bind_comparator_configuration_identity`, `ExecutionEnvironmentRegistry.runtime_lock_sha256`, and `.registry_sha256`.
- Produces: `RunnerAuthority.comparator_tuning`, bound `CompetitionPlan.configurations`, exact configuration-major plan order, and comparator tuning hashes in `CompetitionPlan.input_hashes`.

- [ ] **Step 1: Replace the old cardinality test with an exact block-order test**

```python
def test_tracked_plan_has_exact_2896_rows_and_complete_comparator_blocks() -> None:
    authority = load_runner_authority()
    registry = load_method_registry(ROOT / "study/methods.json")
    bindings = validate_development_manifest_payload(_manifest_payload())
    plan = build_competition_plan(
        registry,
        bindings,
        authority,
        execution_environment_sha256="2" * 64,
        runtime_lock_sha256="1" * 64,
    )
    assert len(plan.entries) == 2_896
    assert len({entry.run_id for entry in plan.entries}) == 2_896
    comparator_rows = [
        entry for entry in plan.entries
        if entry.configuration_kind == "comparator_tuning"
    ]
    assert len(comparator_rows) == 1_632
    assert {entry.method_id for entry in comparator_rows} == {
        "alra", "magic", "dca", "scvi", "saver", "scziva", "afmf",
        "biaeimpute", "sccr", "scsdae",
    }
    assert all(entry.configuration_id != "registry-default" for entry in comparator_rows)
    assert all(entry.configuration_method_identity_sha256 for entry in comparator_rows)
    for configuration in (
        value for value in plan.configurations
        if value.kind == "comparator_tuning"
    ):
        block = [
            entry for entry in comparator_rows
            if entry.method_id == configuration.method_id
            and entry.configuration_id == configuration.configuration_id
        ]
        assert len(block) == 48
        positions = [plan.entries.index(entry) for entry in block]
        assert positions == list(range(min(positions), min(positions) + 48))
    assert len(plan.configurations) == 61
    assert not ({"d3impute", "sctsi"} & {entry.method_id for entry in plan.entries})

    cursor = 0
    for configuration in plan.configurations:
        seeds = (None,) if configuration.method_id == "observed" else (42, 43, 44)
        expected_cells = [
            (binding.dataset_id, seed)
            for binding in bindings
            for seed in seeds
        ]
        block = plan.entries[cursor:cursor + len(expected_cells)]
        assert [(row.dataset_id, row.model_seed) for row in block] == expected_cells
        assert {(row.method_id, row.configuration_id) for row in block} == {
            (configuration.method_id, configuration.configuration_id)
        }
        cursor += len(expected_cells)
    assert cursor == len(plan.entries)

    tuning = authority.comparator_tuning
    assert tuple(
        (row.method_id, row.configuration_id)
        for row in plan.configurations
        if row.kind == "comparator_tuning"
    ) == tuple(
        (row.method_id, row.configuration_id)
        for row in tuning.configurations
    )
    for method_id in tuning.method_order:
        configured = tuning.configurations_for(method_id)
        assert configured[0].is_upstream_default
```

- [ ] **Step 2: Run the plan test red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py::test_tracked_plan_has_exact_2896_rows_and_complete_comparator_blocks \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because the current plan has 1,744 rows and comparator `registry-default` entries.

- [ ] **Step 3: Load comparator authority into `RunnerAuthority`**

Add fields:

```python
    comparator_tuning_file_sha256: str
    comparator_tuning_payload_sha256: str
    comparator_tuning: ComparatorTuningAuthority
    plan_scope: Literal["base_full_panel", "revision_candidate_only"] = "base_full_panel"
```

In `load_runner_authority()`, load the registry once, load the tuning authority against it, require the raw/payload hashes to equal `SelectionAuthority`, add `study/comparator_tuning.json` to the exact file set, and add both hashes plus all 34 configuration payloads to `authority_body`. Keep the 26 MaskImpute/capacity configurations in `RunnerAuthority.configurations`; comparator configurations are bound only after runtime identity is known.

Move the scope invariant forward from Task 13 now. Base authority requires all
26 MaskImpute/capacity rows. The v28 and v29 loaders set
`plan_scope="revision_candidate_only"` and each requires exactly one
MaskImpute candidate with no capacity or comparator configuration. Branch
`RunnerAuthority.__post_init__` on this field. Task 13 completes base-evidence
reuse; it does not redefine runner scope.

- [ ] **Step 4: Bind runtime-dependent comparator configurations and reorder planning**

Change the signature to:

```python
def build_competition_plan(
    registry: MethodRegistry,
    datasets: Sequence[DatasetBinding],
    authority: RunnerAuthority,
    *,
    execution_environment_sha256: str | None = None,
    runtime_lock_sha256: str | None = None,
) -> CompetitionPlan:
```

Require both runtime hashes in production. Build configuration tuples with this exact branch:

```python
    plan_configurations: list[AuthorizedConfiguration] = []
    for spec in registry.methods:
        if spec.execution_scope != "same_input_required":
            continue
        if spec.id == "observed":
            configurations = (AuthorizedConfiguration.registry_default(spec),)
        elif spec.id in {"maskimpute", "capacity-matched-ae"}:
            configurations = authority_by_method.get(spec.id, ())
            if not configurations:
                raise RunnerContractError(f"tracked authority has no configuration for {spec.id}")
        else:
            configurations = tuple(
                AuthorizedConfiguration.from_bound_comparator(
                    bind_comparator_configuration_identity(
                        row,
                        spec,
                        authority.comparator_tuning,
                        runtime_lock_sha256=str(runtime_lock_sha256),
                        environment_registry_sha256=str(execution_environment_sha256),
                    )
                )
                for row in authority.comparator_tuning.configurations_for(spec.id)
            )
            if not configurations:
                raise RunnerContractError(f"comparator tuning has no configuration for {spec.id}")
        plan_configurations.extend(configurations)
```

Replace the current dataset-first nesting with method, configuration, dataset, seed. Preserve the existing preflight-status derivation inside the new loops. Set the new `RunPlanEntry` identity fields from `configuration`. Add both tuning hashes and `runtime_lock_sha256` to `input_hashes`. For `base_full_panel`, assert exact component counts before hashing:

```python
    component_counts = {
        "observed": sum(entry.method_id == "observed" for entry in entries),
        "capacity": sum(entry.method_id == "capacity-matched-ae" for entry in entries),
        "maskimpute": sum(entry.method_id == "maskimpute" for entry in entries),
        "comparators": sum(entry.configuration_kind == "comparator_tuning" for entry in entries),
    }
    if component_counts != {
        "observed": 16, "capacity": 48, "maskimpute": 1_200,
        "comparators": 1_632,
    }:
        raise RunnerContractError("development plan component denominator differs")
```

For `revision_candidate_only`, plan only the one authorized MaskImpute
configuration over the same 16 bindings and seeds `(42, 43, 44)`, and assert
exactly 48 rows. The base cardinality assertion must not run on this scope.

- [ ] **Step 5: Pass the runtime-lock digest from production construction**

In `_run_competition_with_authority`, require `environments.runtime_lock_sha256` and call:

```python
    if environments.runtime_lock_sha256 is None:
        raise RunnerContractError("development runtime lock checksum is absent")
    plan = build_competition_plan(
        registry,
        bindings,
        authority,
        execution_environment_sha256=environments.registry_sha256,
        runtime_lock_sha256=environments.runtime_lock_sha256,
    )
```

- [ ] **Step 6: Run plan, runtime, and prezero schema regressions**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py tests/test_runtime_environments.py \
  tests/test_prezero_evidence.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS with all fixtures updated to carry the three new identity fields; no old 1,744 assertion remains outside archived design documents.

- [ ] **Step 7: Commit the expanded plan**

```bash
git add maskimpute_benchmark/runner.py tests/test_benchmark_runner.py \
  tests/test_runtime_environments.py tests/test_prezero_evidence.py
git commit -m "feat: plan fair comparator development grid"
```

---

### Task 6: Dispatch every comparator with its decoded typed configuration

**Files:**

- Modify: `maskimpute_benchmark/runner.py:5318-5505,6089-6453`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_core_method_adapters.py`
- Modify: `tests/test_priority_method_adapters.py`
- Modify: `tests/test_required_legacy_method_adapters.py`

**Interfaces:**

- Consumes: `decode_comparator_configuration(...)`, `ExecutionRequest.configuration_payload_json`, and its bound identity fields.
- Produces: `RepositoryAdapterDispatcher._comparator_config(request) -> ComparatorAdapterConfig`; every native `run_*` receives `config=decoded` and the payload is re-encoded/rehashed after every return or exception.

- [ ] **Step 1: Add a failing ten-adapter dispatch spy test**

```python
@pytest.mark.parametrize(
    ("method_id", "configuration_id", "field", "expected"),
    (
        ("alra", "alra-default", "k", 0),
        ("magic", "magic-t07", "diffusion_time", 7),
        ("dca", "dca-h32-16-32", "hidden_size", (32, 16, 32)),
        ("scvi", "scvi-z30", "n_latent", 30),
        ("saver", "saver-default", "do_fast", True),
        ("scziva", "scziva-tau-0p05", "tau", 0.05),
        ("afmf", "afmf-sigma-4", "sigma", 4.0),
        ("biaeimpute", "biaeimpute-z256", "latent_size", 256),
        ("sccr", "sccr-k30", "neighbors", 30),
        ("scsdae", "scsdae-zero-0p25", "zero_loss_weight", 0.25),
    ),
)
def test_dispatcher_passes_exact_typed_comparator_config(
    monkeypatch, dispatcher_fixture, request_for_comparator,
    method_id: str, configuration_id: str, field: str, expected: object,
) -> None:
    captured: dict[str, object] = {}

    def fake_adapter(*args, **kwargs):
        captured["config"] = kwargs["config"]
        raise RuntimeError("captured typed comparator config")

    monkeypatch.setattr(
        f"maskimpute_benchmark.methods.run_{method_id}",
        fake_adapter,
    )
    outcome = dispatcher_fixture._execute_validated(
        request_for_comparator(method_id, configuration_id)
    )
    assert outcome.status == "failed"
    assert getattr(captured["config"], field) == expected
```

For `biaeimpute` map the exported function name explicitly because it already matches `run_biaeimpute`; for scZiva use `run_scziva`, and for scSDAE use `run_scsdae`. The fixture must patch the symbol actually imported inside `_execute_validated`, not a stale module alias.

Define both named fixtures in this step. `request_for_comparator` loads the
tracked registry/tuning authority with `require_clean=False`, binds the chosen
row with the fixture runtime-lock/environment hashes, converts it with
`AuthorizedConfiguration.from_bound_comparator`, and calls
`ExecutionRequest.create` with `_method_input()`, seed 42, the exact
method/configuration, and `_authority(maskimpute_ready=True)` extended with the
new tuning fields. `dispatcher_fixture` uses a temporary executable registry
whose requested method points to `Path(sys.executable)` and whose source cache
path is created beneath `tmp_path`; it monkeypatches environment revalidation
only, never request integrity. Add explicit sibling tests named
`test_registry_default_comparator_request_is_rejected` and
`test_dispatcher_revalidates_comparator_payload_after_adapter_attempt`; the
latter replaces the captured frozen dataclass through the test seam and proves
the `finally` re-encode check changes the outcome to `failed`.

- [ ] **Step 2: Run dispatch tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py -k 'typed_comparator_config or registry_default_comparator' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because adapter calls omit `config=` and registry-default comparator requests are accepted.

- [ ] **Step 3: Add the single strict dispatcher decoder**

```python
    @staticmethod
    def _comparator_config(request: ExecutionRequest) -> ComparatorAdapterConfig:
        if request.configuration_kind != "comparator_tuning":
            raise RunnerContractError("comparator request is not tuning-authorized")
        payload = json.loads(
            request.configuration_payload_json,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        config = decode_comparator_configuration(
            request.method_spec.id,
            payload,
            expected_payload_sha256=str(request.configuration_payload_sha256),
        )
        if canonical_sha256(encode_comparator_configuration(config)) != request.configuration_payload_sha256:
            raise RunnerContractError("decoded comparator payload checksum differs")
        return config
```

At the start of the external-comparator branch set `config = self._comparator_config(request)`. Pass `config=config` to ALRA, SAVER, and the eight Python adapter calls. In a `finally` block repeat `encode_comparator_configuration(config)` and compare exact canonical bytes/hash so an adapter-side mutation or wrong dataclass fails even when the adapter raises.

- [ ] **Step 4: Remove development registry-default fallback**

Delete the `execute_competition_plan` branch that reconstructs `AuthorizedConfiguration.registry_default(spec)` when an exact configuration is absent. Add:

```python
        if entry.method_id not in {"observed"} and entry.configuration_id == "registry-default":
            raise RunnerContractError("publication comparator cannot use registry-default")
```

- [ ] **Step 5: Run dispatcher and adapter contract tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py tests/test_core_method_adapters.py \
  tests/test_priority_method_adapters.py tests/test_required_legacy_method_adapters.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS; tests whose real external environments are absent may retain their pre-existing skips.

- [ ] **Step 6: Commit typed dispatch**

```bash
git add maskimpute_benchmark/runner.py tests/test_benchmark_runner.py \
  tests/test_core_method_adapters.py tests/test_priority_method_adapters.py \
  tests/test_required_legacy_method_adapters.py
git commit -m "feat: dispatch typed comparator configurations"
```

---

### Task 7: Make budget replay exact and incomplete grids unselectable

**Files:**

- Modify: `maskimpute_benchmark/runner.py:70-74,2598-2728,3701-5317,5318-5505`
- Modify: `tests/test_benchmark_runner.py`

**Interfaces:**

- Consumes: plan configuration kinds/statuses and checkpoint records.
- Produces: `_counts_toward_configuration_limit(entry) -> bool`, `replay_development_budget(...) -> DevelopmentBudget`, exact checkpoint-ledger validation, and explicit comparator-selection blockers.

- [ ] **Step 1: Write failing shared-ledger and tamper tests**

```python
def test_comparator_configs_share_method_budget_and_restore_exactly(
    magic_spec, completed_outcome
) -> None:
    budget = DevelopmentBudget()
    hashes = tuple(f"{value:064x}" for value in range(1, 5))
    for digest in hashes:
        assert budget.authorize(magic_spec, digest).authorized
        budget.record(magic_spec, digest, completed_outcome)
    restored = DevelopmentBudget()
    for digest in hashes:
        restored.restore(magic_spec, digest, "completed", completed_outcome.runtime_seconds)
    assert restored.to_dict() == budget.to_dict()


def test_checkpoint_rejects_coherently_rehashed_budget_tamper(
    completed_checkpoint_fixture,
) -> None:
    checkpoint_path, plan, prepared = completed_checkpoint_fixture
    payload = json.loads(checkpoint_path.read_text())
    payload["budget"]["magic"]["consumed_seconds"] += 1
    payload["checkpoint_sha256"] = canonical_sha256({
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    })
    checkpoint_path.write_bytes(runner_module._canonical_bytes(payload) + b"\n")
    with pytest.raises(RunnerContractError, match="budget ledger differs from replay"):
        CheckpointStore(checkpoint_path.parent).load(
            plan,
            registry=load_method_registry(METHODS_PATH),
            prepared_datasets=prepared,
        )
```

Implement the named fixtures in `tests/test_benchmark_runner.py` rather than
assuming global fixtures: derive `magic_spec` from `load_method_registry`, make
`completed_outcome` with `AdapterOutcome.completed` and the existing observed
snapshot helper, and build `completed_checkpoint_fixture` with the existing
`_two_method_plan`, `_prepared_truth_dataset`, `CheckpointStore`, and fake
executor. Serialize the coherently rehashed checkpoint with the production
`_canonical_bytes(payload) + b"\n"` helper. After Step 3, every `load` call in
this test includes `registry=load_method_registry(METHODS_PATH)`.

- [ ] **Step 2: Run budget tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py -k 'share_method_budget or budget_tamper or incomplete_grid' \
  -q -W error -p no:cacheprovider
```

Expected: the tampered ledger is currently accepted and comparator kinds do not count toward the configuration ledger.

- [ ] **Step 3: Centralize configuration-limit and ledger replay policy**

```python
def _counts_toward_configuration_limit(entry: RunPlanEntry) -> bool:
    return entry.configuration_kind in {"candidate_search", "comparator_tuning"}


def _budget_scope(entry: RunPlanEntry) -> str:
    return (
        f"{entry.method_id}:{entry.configuration_kind}"
        if entry.method_id == "maskimpute"
        else entry.method_id
    )


def replay_development_budget(
    registry: MethodRegistry,
    entries: Sequence[RunPlanEntry],
    records: Sequence[Mapping[str, object]],
) -> DevelopmentBudget:
    budget = DevelopmentBudget()
    for entry, stored in zip(entries, records, strict=False):
        run = stored.get("run")
        if not isinstance(run, Mapping):
            raise RunnerContractError("checkpoint budget replay record is invalid")
        budget.restore(
            registry.by_id(entry.method_id),
            entry.configuration_sha256,
            str(run.get("status")),
            run.get("runtime_seconds"),
            counts_toward_configuration_limit=_counts_toward_configuration_limit(entry),
            budget_scope=_budget_scope(entry),
        )
    return budget
```

Use these helpers for resume, authorization, and recording. Choose the exact
interface `CheckpointStore.load(plan, *, registry, prepared_datasets)` and
update every caller. Before `zip`, reject `len(records) > len(entries)`; then
replay the complete stored prefix and require
`payload["budget"] == replay.to_dict()` before returning. Do not add a second
plan-owned ledger representation.

- [ ] **Step 4: Keep non-scientific incomplete statuses as hard blockers**

Use the exact sets:

```python
INTRINSIC_TERMINAL_STATUSES = frozenset({
    "failed", "timeout", "resource_exceeded", "unavailable",
})
COMPARATOR_SELECTION_BLOCKING_STATUSES = frozenset({
    "budget_exhausted", "blocked_authority", "infrastructure_error",
})
```

Persisted infrastructure errors remain records and resume begins after them; only a transaction intent without a durable record is recovered/retried. Add a checkpoint-level `comparator_selection_status` summary derived from records, never caller supplied. It is `complete_terminal_denominator` only when every comparator-tuning entry is completed or intrinsic terminal and `blocked_incomplete_denominator` otherwise.

- [ ] **Step 5: Run budget, checkpoint, and transaction recovery tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py -k 'budget or checkpoint or transaction or infrastructure' \
  -q -W error -p no:cacheprovider
```

Expected: PASS, including no selective retry of any persisted terminal record.

- [ ] **Step 6: Commit budget completeness**

```bash
git add maskimpute_benchmark/runner.py tests/test_benchmark_runner.py
git commit -m "fix: replay comparator development budgets exactly"
```

---

### Task 8: Bound retained artifacts and fail storage preflight before any write

**Files:**

- Modify: `maskimpute_benchmark/runner.py:3805-5317,6720-6758`
- Modify: `maskimpute_benchmark/prezero_evidence.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_prezero_evidence.py`

**Interfaces:**

- Consumes: exact plan entries, prepared retained shapes, `zlib_compress_bound`, and the Task 1 development ceilings.
- Produces: `DevelopmentStoragePreflight`, `development_storage_preflight(...)`, `require_development_storage_capacity(...)`, hash-only log receipts, a bounded executor receipt, bounded records/checkpoints, and a storage receipt bound into the checkpoint.

- [ ] **Step 1: Add failing arithmetic and zero-write tests**

```python
def test_development_storage_preflight_matches_exact_plan_bound(
    production_plan, prepared_development_panel
) -> None:
    receipt = development_storage_preflight(
        production_plan,
        prepared_development_panel,
        completed_records=0,
    )
    matrix_bytes = sum(
        2 * prepared_development_panel[entry.dataset_id].method_input.counts.nbytes
        for entry in production_plan.entries
        if entry.preflight_status == "planned"
    )
    score_bytes = sum(
        zlib_compress_bound(
            prepared_development_panel[entry.dataset_id].method_input.counts.nbytes
        )
        for entry in production_plan.entries
        if entry.preflight_status == "planned"
        and entry.method_id == "maskimpute"
        and entry.requires_count_score
    )
    executable_count = sum(
        entry.preflight_status == "planned" for entry in production_plan.entries
    )
    expected = (
        matrix_bytes + score_bytes
        + executable_count * (
            2 * DEVELOPMENT_MAX_LOG_RECEIPT_BYTES
            + DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES
        )
        + len(production_plan.entries) * DEVELOPMENT_MAX_RECORD_BYTES
        + DEVELOPMENT_MAX_CHECKPOINT_BYTES + DEVELOPMENT_STORAGE_RESERVE_BYTES
    )
    assert receipt.required_free_bytes == expected
    assert 20 * 1024**3 <= expected <= 27 * 1024**3


def test_storage_failure_occurs_before_output_directory_creation(
    tmp_path, production_plan, prepared_development_panel
) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(RunnerContractError, match="insufficient development storage"):
        require_development_storage_capacity(
            output,
            production_plan,
            prepared_development_panel,
            completed_records=0,
            available_bytes=1,
        )
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
```

Define `production_plan` in this module from
`validate_development_manifest_payload(_manifest_payload())`, the tracked
registry, a synthetic copy of the Task 5 base authority whose count-score and
calibration bindings are valid fixture hashes (so all 2,896 rows are planned),
and fixture runtime hashes. Define
`prepared_development_panel` by constructing one `PreparedDataset` per binding
with an immutable 900-by-500 float64 `MethodInput`; use the binding's exact
dataset/hash fields and reuse the same zero-filled byte buffer because only
shape and retained byte bounds matter. Assert the fixture itself has 16 keys,
all shapes `(900, 500)`, and a 2,896-row plan before using it in arithmetic.

- [ ] **Step 2: Run storage tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py -k 'storage_preflight or output_directory_creation' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because no development preflight API exists.

- [ ] **Step 3: Implement pure bound calculation and fail-before-write check**

```python
@dataclass(frozen=True, slots=True)
class DevelopmentStoragePreflight:
    schema: str
    planned_run_count: int
    completed_record_count: int
    remaining_executable_count: int
    matrix_bytes: int
    prezero_zlib_bound_bytes: int
    log_receipt_bytes: int
    executor_receipt_bytes: int
    record_bytes: int
    checkpoint_bytes: int
    reserve_bytes: int
    required_free_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def development_storage_preflight(
    plan: CompetitionPlan,
    prepared_datasets: Mapping[str, PreparedDataset],
    *,
    completed_records: int,
) -> DevelopmentStoragePreflight:
    if type(completed_records) is not int or not 0 <= completed_records <= len(plan.entries):
        raise RunnerContractError("completed storage record count is invalid")
    remaining = plan.entries[completed_records:]
    executable = tuple(entry for entry in remaining if entry.preflight_status == "planned")
    matrix_bytes = sum(
        2 * prepared_datasets[entry.dataset_id].method_input.counts.nbytes
        for entry in executable
    )
    prezero_bytes = sum(
        zlib_compress_bound(
            prepared_datasets[entry.dataset_id].method_input.counts.nbytes
        )
        for entry in executable
        if entry.method_id == "maskimpute" and entry.requires_count_score
    )
    log_bytes = 2 * len(executable) * DEVELOPMENT_MAX_LOG_RECEIPT_BYTES
    executor_bytes = len(executable) * DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES
    record_bytes = len(remaining) * DEVELOPMENT_MAX_RECORD_BYTES
    required = (
        matrix_bytes + prezero_bytes + log_bytes + executor_bytes + record_bytes
        + DEVELOPMENT_MAX_CHECKPOINT_BYTES + DEVELOPMENT_STORAGE_RESERVE_BYTES
    )
    return DevelopmentStoragePreflight(
        schema="maskimpute-development-storage-preflight-v1",
        planned_run_count=len(plan.entries),
        completed_record_count=completed_records,
        remaining_executable_count=len(executable),
        matrix_bytes=matrix_bytes,
        prezero_zlib_bound_bytes=prezero_bytes,
        log_receipt_bytes=log_bytes,
        executor_receipt_bytes=executor_bytes,
        record_bytes=record_bytes,
        checkpoint_bytes=DEVELOPMENT_MAX_CHECKPOINT_BYTES,
        reserve_bytes=DEVELOPMENT_STORAGE_RESERVE_BYTES,
        required_free_bytes=required,
    )


def require_development_storage_capacity(
    output_dir: Path,
    plan: CompetitionPlan,
    prepared_datasets: Mapping[str, PreparedDataset],
    *,
    completed_records: int,
    available_bytes: int | None = None,
) -> DevelopmentStoragePreflight:
    receipt = development_storage_preflight(
        plan, prepared_datasets, completed_records=completed_records
    )
    probe = output_dir.absolute()
    while not probe.exists():
        if probe.parent == probe:
            raise RunnerContractError("development storage filesystem is unavailable")
        probe = probe.parent
    observed = (
        int(os.statvfs(probe).f_bavail * os.statvfs(probe).f_frsize)
        if available_bytes is None else available_bytes
    )
    if observed < receipt.required_free_bytes:
        raise RunnerContractError("insufficient development storage before scientific write")
    return receipt
```

Compute and bind one immutable full-plan receipt with `completed_records=0` into
the plan/checkpoint input hashes. Its hash binds policy, all retained shapes,
and the worst-case full-plan bound and therefore never changes on resume.

For resume, add `CheckpointStore.inspect_prefix(plan, *, registry,
prepared_datasets) -> int`. It opens and validates the checkpoint envelope,
plan hash, record prefix, budget replay, and artifact bindings read-only; it
must not recover a transaction, create a directory, unlink, or rewrite. Use the
returned count to compute a second remaining-capacity receipt, require free
space, and only then call normal `load`, transaction recovery, or
`_ensure_root`. The remaining receipt is retained in the next checkpoint for
audit but is not an immutable plan input. Add a regression with an interrupted
transaction and insufficient free space proving inspection makes zero writes
or deletions.

- [ ] **Step 4: Replace raw log persistence with bounded hash-only receipts**

Add:

```python
def _stream_receipt(stream: str, raw: bytes) -> bytes:
    if stream not in {"stdout", "stderr"} or type(raw) is not bytes:
        raise RunnerContractError("execution stream receipt input is invalid")
    value = {
        "schema": "maskimpute-development-stream-receipt-v1",
        "stream": stream,
        "raw_byte_count": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "retention": "hash_only_no_raw_bytes_or_private_paths",
    }
    encoded = _canonical_bytes(value) + b"\n"
    if len(encoded) > DEVELOPMENT_MAX_LOG_RECEIPT_BYTES:
        raise RunnerContractError("execution stream receipt exceeds its bound")
    return encoded


def _executor_receipt(attempt: EvaluatedAttempt) -> bytes:
    value = {
        "schema": "maskimpute-development-executor-receipt-v1",
        "run_id": attempt.run.run_id,
        "status": attempt.run.status,
        "reason": attempt.run.reason,
        "runtime_seconds": attempt.run.runtime_seconds,
        "peak_rss_bytes": attempt.run.peak_rss_bytes,
        "peak_gpu_bytes": attempt.run.peak_gpu_bytes,
        "rss_measurement": attempt.run.rss_measurement,
        "gpu_measurement": attempt.run.gpu_measurement,
        "stdout_sha256": hashlib.sha256(attempt.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(attempt.stderr).hexdigest(),
    }
    encoded = _canonical_bytes(value) + b"\n"
    if len(encoded) > DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES:
        raise RunnerContractError("executor receipt exceeds its bound")
    return encoded
```

Publish `.stdout.json`, `.stderr.json`, and `.executor.json`; retain their file hashes and the original stream hashes/byte counts in the run record. Never persist `attempt.stdout` or `attempt.stderr`. Add the executor artifact to transaction-intent allowed paths and verify canonical receipt semantics on load.

- [ ] **Step 5: Enforce record and checkpoint byte ceilings**

Before publishing a transaction record require:

```python
        record_raw = _canonical_bytes(staged_record) + b"\n"
        if len(record_raw) > DEVELOPMENT_MAX_RECORD_BYTES:
            raise RunnerContractError("development record exceeds its byte bound")
```

In `_publish_checkpoint` and `_read_checkpoint_bytes` reject payloads/files larger than `DEVELOPMENT_MAX_CHECKPOINT_BYTES` before allocating or writing. Verify the executor/log receipts before loading matrix artifacts.

- [ ] **Step 6: Run storage, checkpoint, recovery, and prezero tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py tests/test_prezero_evidence.py \
  -k 'storage or checkpoint or transaction or stream or executor or prezero' \
  -q -W error -p no:cacheprovider
```

Expected: PASS; fixtures assert raw stderr/private absolute paths never occur in retained bytes.

- [ ] **Step 7: Commit bounded development evidence**

```bash
git add maskimpute_benchmark/runner.py maskimpute_benchmark/prezero_evidence.py \
  tests/test_benchmark_runner.py tests/test_prezero_evidence.py
git commit -m "feat: preflight bounded development evidence"
```

---

### Task 9: Add the fixed truth-free comparator smoke gate

**Files:**

- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Modify: `maskimpute_benchmark/runner.py:6089-6453,6641-6758`
- Create: `scripts/run_comparator_tuning_smoke.py`
- Modify: `tests/test_comparator_tuning.py`
- Modify: `tests/test_benchmark_runner.py`

**Interfaces:**

- Consumes: all 34 bound configurations, `RepositoryAdapterDispatcher`, runtime locks, and the tracked smoke policy.
- Produces: `build_comparator_smoke_input() -> MethodInput`, `build_comparator_smoke_receipt(...) -> dict[str, object]`, `load_comparator_smoke_receipt(repository, authority, registry)`, `run_comparator_tuning_smoke(repository) -> Mapping[str, object]`, and a required smoke hash in the base production plan.

- [ ] **Step 1: Add failing fixture and projection tests**

```python
def test_smoke_input_is_exact_truth_free_900_by_500() -> None:
    method_input = build_comparator_smoke_input()
    assert method_input.shape == (900, 500)
    assert method_input.counts[0, 0] == 0
    assert method_input.counts[17, 31] == (
        17 * 17 + 31 * 31 + 7 * (17 ^ 31)
    ) % 6
    assert len(method_input.obs_covariates) == 1
    batch = method_input.obs_covariates[0]
    assert batch.name == "batch"
    assert batch.categories == ("batch-0", "batch-1")
    assert batch.values == tuple(f"batch-{index % 2}" for index in range(900))
    assert not hasattr(method_input, "truth")


def test_smoke_receipt_requires_all_34_completed_and_projected_budget(
    complete_smoke_outcomes, smoke_authority, smoke_registry, smoke_bound_rows,
) -> None:
    receipt = build_comparator_smoke_receipt(
        complete_smoke_outcomes,
        authority=smoke_authority,
        registry=smoke_registry,
        bound_configurations=smoke_bound_rows,
        runtime_lock_sha256="3" * 64,
        environment_registry_sha256="4" * 64,
    )
    assert receipt["planned_configuration_count"] == 34
    assert receipt["completed_configuration_count"] == 34
    assert receipt["projection_multiplier"] == 48
    assert receipt["status"] == "ready"
    broken = list(complete_smoke_outcomes)
    broken[0] = replace(broken[0], status="unavailable", reason="smoke_unavailable")
    with pytest.raises(ComparatorTuningError, match="all configurations must complete"):
        build_comparator_smoke_receipt(
            broken,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
            runtime_lock_sha256="3" * 64,
            environment_registry_sha256="4" * 64,
        )
```

Define the four fixtures directly in `tests/test_comparator_tuning.py`:
`smoke_registry` loads `study/methods.json`; `smoke_authority` loads the tuning
authority with `require_clean=False`; `smoke_bound_rows` binds every authority
row in order using runtime hash `"3" * 64` and environment hash `"4" * 64`;
`complete_smoke_outcomes` maps those rows one-for-one to
`ComparatorSmokeOutcome(status="completed", reason=None,
runtime_seconds=1.0, peak_rss_bytes=1024, peak_gpu_bytes=0,
rss_measurement="fixed_test_sampler", gpu_measurement="fixed_test_sampler")`.
Assert the fixture pair/identity tuple equals the authority tuple before the
receipt test. This is the sole smoke test-data factory; do not introduce a
second implicit configuration list.

- [ ] **Step 2: Run smoke tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'smoke_input or smoke_receipt' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because the smoke APIs are absent.

- [ ] **Step 3: Implement deterministic input generation and receipt types**

Add `import numpy as np`; `hashlib` and `asdict` are already imported by Task 1.

```python
@dataclass(frozen=True, slots=True)
class ComparatorSmokeOutcome:
    method_id: str
    configuration_id: str
    configuration_payload_sha256: str
    configuration_method_identity_sha256: str
    status: str
    reason: str | None
    runtime_seconds: float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    rss_measurement: str
    gpu_measurement: str


def build_comparator_smoke_input() -> MethodInput:
    from .methods.base import CovariateColumn, MethodInput
    counts = np.fromfunction(
        lambda cell, gene: (
            17 * cell.astype(np.int64)
            + 31 * gene.astype(np.int64)
            + 7 * np.bitwise_xor(cell.astype(np.int64), gene.astype(np.int64))
        ) % 6,
        (900, 500),
        dtype=np.int64,
    ).astype(np.float64)
    count_bytes = np.asarray(counts, dtype="<f8", order="C").tobytes(order="C")
    batch_values = tuple(f"batch-{index % 2}" for index in range(900))
    return MethodInput(
        source_dataset_sha256=canonical_sha256({
            "schema": "maskimpute-comparator-smoke-source-v1",
            "cells": 900,
            "genes": 500,
            "formula": "(17*cell+31*gene+7*(cell^gene))%6",
            "batch_rule": "alternating_batch-0_batch-1",
        }),
        obs_ids=tuple(f"smoke-cell-{index:04d}" for index in range(900)),
        var_ids=tuple(f"smoke-gene-{index:04d}" for index in range(500)),
        shape=(900, 500),
        obs_covariates=(CovariateColumn(
            name="batch",
            kind="categorical",
            dtype="category",
            values=batch_values,
            categories=("batch-0", "batch-1"),
            ordered=False,
            codes=tuple(index % 2 for index in range(900)),
        ),),
        var_covariates=(),
        _count_bytes=count_bytes,
        _normalization_bytes=b'"raw_counts"',
    )


def comparator_smoke_input_sha256(method_input: MethodInput) -> str:
    return canonical_sha256({
        "schema": "maskimpute-comparator-smoke-method-input-v1",
        "source_dataset_sha256": method_input.source_dataset_sha256,
        "obs_ids": list(method_input.obs_ids),
        "var_ids": list(method_input.var_ids),
        "shape": list(method_input.shape),
        "obs_covariates": [asdict(row) for row in method_input.obs_covariates],
        "var_covariates": [asdict(row) for row in method_input.var_covariates],
        "counts_sha256": hashlib.sha256(
            np.asarray(method_input.counts, dtype="<f8", order="C").tobytes(order="C")
        ).hexdigest(),
        "normalization": method_input.normalization,
    })


def build_comparator_smoke_receipt(
    outcomes: Sequence[ComparatorSmokeOutcome],
    *,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
    bound_configurations: Sequence[BoundComparatorConfiguration],
    runtime_lock_sha256: str,
    environment_registry_sha256: str,
) -> dict[str, object]:
    rows = tuple(outcomes)
    expected = tuple(
        (
            row.configuration.method_id,
            row.configuration.configuration_id,
            row.configuration_payload_sha256,
            row.configuration_method_identity_sha256,
        )
        for row in bound_configurations
    )
    observed = tuple(
        (
            row.method_id, row.configuration_id,
            row.configuration_payload_sha256,
            row.configuration_method_identity_sha256,
        )
        for row in rows
    )
    if len(expected) != 34 or observed != expected:
        raise ComparatorTuningError("smoke denominator differs from bound authority order")
    if any(row.status != "completed" or row.reason is not None for row in rows):
        raise ComparatorTuningError("all configurations must complete before scientific execution")
    if any(
        not math.isfinite(row.runtime_seconds)
        or row.runtime_seconds < 0
        or type(row.peak_rss_bytes) is not int
        or row.peak_rss_bytes < 0
        or type(row.peak_gpu_bytes) is not int
        or row.peak_gpu_bytes < 0
        or not row.rss_measurement
        or not row.gpu_measurement
        for row in rows
    ):
        raise ComparatorTuningError("smoke measurement is invalid")
    projected: dict[str, float] = {}
    for row in rows:
        projected[row.method_id] = projected.get(row.method_id, 0.0) + 48.0 * row.runtime_seconds
        if row.peak_rss_bytes > 48 * 1024**3 or row.peak_gpu_bytes > 14 * 1024**3:
            raise ComparatorTuningError("smoke resource cap is exceeded")
    if any(
        seconds > (
            8 * 3600 if registry.by_id(method).resources.gpu_required else 24 * 3600
        )
        for method, seconds in projected.items()
    ):
        raise ComparatorTuningError("projected comparator grid exceeds its method budget")
    unsigned = {
        "schema_version": 1,
        "artifact_type": "maskimpute-comparator-smoke-receipt-v1",
        "scope": "fixed_nonstudy_truth_free_operational_feasibility",
        "tuning_authority_file_sha256": authority.file_sha256,
        "tuning_authority_payload_sha256": authority.payload_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "environment_registry_sha256": environment_registry_sha256,
        "fixture_sha256": comparator_smoke_input_sha256(
            build_comparator_smoke_input()
        ),
        "model_seed": 42,
        "projection_multiplier": 48,
        "planned_configuration_count": 34,
        "completed_configuration_count": 34,
        "status": "ready",
        "projected_method_runtime_seconds": dict(sorted(projected.items())),
        "outcomes": [asdict(row) for row in rows],
        "output_retention": "discarded_without_evaluator_or_metrics",
    }
    return {**unsigned, "payload_sha256": canonical_sha256(unsigned)}
```

Adjust `MethodInput` construction to its exact existing constructor fields; do not add truth, evaluator, or score fields.

- [ ] **Step 4: Execute only the smoke adapter boundary and discard outputs**

Implement `run_comparator_tuning_smoke(repository)` by loading the fixed registry, tuning authority, runtime lock/environment registry, binding all 34 identities, building one `ExecutionRequest` per configuration with seed 42 and smoke identity fields, and calling the spawned dispatcher. Do not call `evaluate_adapter_outcome`, `CheckpointStore`, or any evaluator. Convert each `AdapterOutcome` immediately into `ComparatorSmokeOutcome`, drop `outcome.execution`, stdout, and stderr, build the receipt, publish it create-only/atomic at the tracked fixed path, and revalidate identical bytes.

The only test seam is a private keyword-only `_executor` argument defaulting to the production spawned dispatcher; the public script never exposes it.

- [ ] **Step 5: Add the no-override script**

Create `scripts/run_comparator_tuning_smoke.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maskimpute_benchmark.comparator_tuning import (  # noqa: E402
    ComparatorTuningError,
    run_comparator_tuning_smoke,
)


def main() -> int:
    argparse.ArgumentParser(
        description="Run the fixed truth-free 34-configuration comparator smoke gate."
    ).parse_args()
    try:
        receipt = run_comparator_tuning_smoke(ROOT)
    except (ComparatorTuningError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "path": "artifacts/study/development/evaluation/comparator_smoke.json",
        "payload_sha256": receipt["payload_sha256"],
        "status": receipt["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Make base production execution require the validated smoke receipt**

At the start of `run_development_competition`, call `load_comparator_smoke_receipt`; add its raw-file and payload hashes to a replaced `RunnerAuthority` and therefore `CompetitionPlan.input_hashes`. Revision runners inherit those hashes. If the receipt is absent, invalid, non-ready, or differs from tuning/runtime identities, raise before storage preflight and before output creation. Tests of pure planning may pass a fixture receipt explicitly; production has no bypass.

- [ ] **Step 7: Run smoke, CLI, and runner-gate tests without real adapters**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py tests/test_benchmark_runner.py \
  -k 'smoke or scientific_execution_requires_smoke' \
  -q -W error -p no:cacheprovider
```

Expected: PASS using injected deterministic fake outcomes; the real smoke script is not executed by tests.

- [ ] **Step 8: Commit the smoke gate**

```bash
git add maskimpute_benchmark/comparator_tuning.py maskimpute_benchmark/runner.py \
  scripts/run_comparator_tuning_smoke.py tests/test_comparator_tuning.py \
  tests/test_benchmark_runner.py
git commit -m "feat: gate competition on comparator smoke"
```

---

### Task 10: Implement exact seed-view-draw collapse, Pareto filtering, and quarter-ranks

**Files:**

- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Modify: `tests/test_comparator_tuning.py`

**Interfaces:**

- Consumes: validated comparator-tuning checkpoint records and the six fixed metrics.
- Produces: `CollapsedComparatorConfiguration`, `collapse_comparator_configuration(...)`, `pareto_configuration_ids(...)`, `metric_rank_quarters(...)`, and `select_one_comparator_method(...)`.

- [ ] **Step 1: Add failing golden tests for nesting, dominance, ties, and tuple order**

```python
def test_seed_view_draw_collapse_and_quarter_rank_golden() -> None:
    records = golden_comparator_records(
        method_id="magic",
        configuration_values={
            "magic-t03": (1.0, 4.0, 2.0, 3.0, 2.0, 4.0),
            "magic-t01": (2.0, 2.0, 2.0, 2.0, 3.0, 2.0),
            "magic-t05": (3.0, 1.0, 3.0, 1.0, 1.0, 3.0),
            "magic-t07": (4.0, 5.0, 4.0, 5.0, 4.0, 5.0),
        },
        duplicate_each_seed_value=True,
    )
    result = select_one_comparator_method("magic", records, golden_authority())
    assert result.configuration_ids == (
        "magic-t03", "magic-t01", "magic-t05", "magic-t07"
    )
    assert result.eligible_configuration_ids == result.configuration_ids
    assert result.pareto_configuration_ids == (
        "magic-t03", "magic-t01", "magic-t05"
    )
    assert result.configuration("magic-t03").unit_counts == {
        "mse": 8,
        "mse_dropout": 8,
        "gnrmse": 8,
        "mse_pre_dropout_zero": 2,
        "corr_err": 8,
        "mse_non_dropout_nonzero": 8,
    }
    assert all(
        type(value) is int
        for row in result.pareto_rows
        for value in row.metric_rank_quarters.values()
    )
    assert result.selected_configuration_id == min(
        result.pareto_rows,
        key=lambda row: row.selection_tuple,
    ).configuration_id


def test_average_ties_encode_exact_quarter_rank_integer() -> None:
    assert metric_rank_quarters({
        "a": (1.0, 1.0),
        "b": (1.0, 2.0),
        "c": (3.0, 2.0),
    }) == {"a": 5, "b": 8, "c": 11}
```

The helper creates all 16 datasets, both views per draw, three seeds, and the applicable metric rows. Duplicating identical seed rows must not change the eight/two independent-unit counts.

Implement `golden_authority()` as the tracked tuning authority loaded with the
tracked registry and `require_clean=False`. Implement
`golden_comparator_records` in the same test module from the canonical ordered
cartesian product `mechanism in (symsim, sergio, sparsim, semisynthetic)`,
`draw in (draw-01, draw-02)`, `view in (moderate, severe)`, and seed in
`(42, 43, 44)`. Each emitted record carries the selected authority row's exact
payload/method identity, `status="completed"`, and one metric row per
panel-wide metric; emit `mse_pre_dropout_zero` only for SymSim. Derive the
configured six values deterministically by metric position and add a constant
draw/view offset shared by configurations so the expected Pareto relations are
unchanged. Assert the helper emits exactly 48 run records and 252 metric rows
per configuration (240 panel-wide plus 12 SymSim prezero) before selection.

- [ ] **Step 2: Run selection-math tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'collapse or quarter_rank or pareto' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because collapse/rank APIs are absent.

- [ ] **Step 3: Implement exact average ranks and median quarter encoding**

```python
def _average_rank_twice(target: float, values: Sequence[float]) -> int:
    below = sum(value < target for value in values)
    tied = sum(value == target for value in values)
    return 2 * below + tied + 1


def metric_rank_quarters(
    unit_values: Mapping[str, Sequence[float]],
) -> dict[str, int]:
    ids = tuple(unit_values)
    if not ids or len({len(tuple(unit_values[item])) for item in ids}) != 1:
        raise ComparatorTuningError("rank unit denominator differs")
    ranks_twice: dict[str, list[int]] = {item: [] for item in ids}
    unit_count = len(tuple(unit_values[ids[0]]))
    for unit in range(unit_count):
        values = [float(unit_values[item][unit]) for item in ids]
        if not all(math.isfinite(value) for value in values):
            raise ComparatorTuningError("rank value is nonfinite")
        for item, value in zip(ids, values, strict=True):
            ranks_twice[item].append(_average_rank_twice(value, values))
    result: dict[str, int] = {}
    for item, ranks in ranks_twice.items():
        ordered = sorted(ranks)
        middle = len(ordered) // 2
        result[item] = (
            2 * ordered[middle]
            if len(ordered) % 2
            else ordered[middle - 1] + ordered[middle]
        )
    return result
```

- [ ] **Step 4: Implement collapse and Pareto semantics**

Use immutable rows with exact fields:

```python
SELECTION_METRICS = (
    "mse", "mse_dropout", "gnrmse", "mse_pre_dropout_zero",
    "corr_err", "mse_non_dropout_nonzero",
)


@dataclass(frozen=True, slots=True)
class CollapsedComparatorConfiguration:
    method_id: str
    configuration_id: str
    configuration_payload_sha256: str
    configuration_method_identity_sha256: str
    eligible: bool
    eligibility_reason: str | None
    status_counts: Mapping[str, int]
    reason_histogram: Mapping[str, int]
    unit_ids: Mapping[str, tuple[str, ...]]
    unit_values: Mapping[str, tuple[float, ...]]
    unit_counts: Mapping[str, int]
    metric_medians: Mapping[str, float]


def pareto_configuration_ids(
    rows: Sequence[CollapsedComparatorConfiguration],
) -> tuple[str, ...]:
    eligible = tuple(row for row in rows if row.eligible)
    retained: list[str] = []
    for target in eligible:
        dominated = any(
            other.configuration_id != target.configuration_id
            and all(other.metric_medians[name] <= target.metric_medians[name] for name in SELECTION_METRICS)
            and any(other.metric_medians[name] < target.metric_medians[name] for name in SELECTION_METRICS)
            for other in eligible
        )
        if not dominated:
            retained.append(target.configuration_id)
    return tuple(retained)
```

`collapse_comparator_configuration` must require exactly 48 unique dataset/seed cells; require completed status for eligibility; average the three seeds within each dataset; pair moderate/severe by `(mechanism, biological_id)`; emit eight ordered unit IDs/values for five metrics and only the two ordered SymSim unit IDs/values for prezero. Before ranking, require every Pareto configuration to expose the identical canonical unit-ID tuple for a metric. A malformed missing/duplicate metric row, nonfinite numeric value, wrong applicability, or identity drift raises. A canonical metric row with `status` intrinsic-terminal and `value=None` instead keeps the configuration in the audit denominator and makes it ineligible; it must not abort selection of the method's other configurations.

- [ ] **Step 5: Implement the exact deterministic tuple**

```python
@dataclass(frozen=True, slots=True)
class RankedComparatorConfiguration:
    configuration_id: str
    metric_rank_quarters: Mapping[str, int]
    selection_tuple: tuple[int, int, int, int, int, int, int, int, int, str]


def _ranked_pareto_rows(
    rows: Sequence[CollapsedComparatorConfiguration],
    defaults: Mapping[str, bool],
) -> tuple[RankedComparatorConfiguration, ...]:
    pareto_ids = pareto_configuration_ids(rows)
    by_id = {row.configuration_id: row for row in rows}
    metric_ranks = {
        metric: metric_rank_quarters({
            item: by_id[item].unit_values[metric] for item in pareto_ids
        })
        for metric in SELECTION_METRICS
    }
    result = []
    for item in pareto_ids:
        ranks = tuple(metric_ranks[metric][item] for metric in SELECTION_METRICS)
        result.append(RankedComparatorConfiguration(
            configuration_id=item,
            metric_rank_quarters=dict(zip(SELECTION_METRICS, ranks, strict=True)),
            selection_tuple=(
                max(ranks), sum(ranks), *ranks,
                0 if defaults[item] else 1, item,
            ),
        ))
    return tuple(result)
```

- [ ] **Step 6: Run all pure selection tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'collapse or pareto or rank or selection_tuple' \
  -q -W error -p no:cacheprovider
```

Expected: PASS with exact integer quarter-ranks and stable selected IDs.

- [ ] **Step 7: Commit pure comparator selection math**

```bash
git add maskimpute_benchmark/comparator_tuning.py tests/test_comparator_tuning.py
git commit -m "feat: select comparator configurations exactly"
```

---

### Task 11: Publish and validate the create-only comparator-selection receipt

**Files:**

- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Create: `scripts/select_comparator_configurations.py`
- Modify: `tests/test_comparator_tuning.py`

**Interfaces:**

- Consumes: exact 2,896-row completed checkpoint/plan, pure method selection from Task 10, registry/control status, and tuning authority.
- Produces: `ComparatorReadiness`, `build_comparator_selection_receipt(...)`, `publish_comparator_selection(repository)`, and `load_comparator_selection_receipt(repository, *, expected_checkpoint=None)`.

- [ ] **Step 1: Write failing readiness, independence, and publication tests**

```python
def test_readiness_requires_controls_established_and_three_modern(
    complete_selection_fixture,
) -> None:
    receipt = build_comparator_selection_receipt(**complete_selection_fixture)
    assert receipt["readiness"]["status"] == "ready"
    assert receipt["readiness"]["modern_selectable_count"] == 4

    three_modern = selection_fixture_with_intrinsic_unavailable(
        complete_selection_fixture, "biaeimpute"
    )
    receipt = build_comparator_selection_receipt(**three_modern)
    assert receipt["readiness"]["status"] == "ready"
    assert receipt["methods"]["biaeimpute"]["selected_configuration_id"] is None
    assert len(receipt["methods"]["biaeimpute"]["nonexecution_identity_sha256"]) == 64

    with pytest.raises(ComparatorTuningError, match="publication readiness"):
        build_comparator_selection_receipt(**selection_fixture_with_intrinsic_unavailable(
            three_modern, "sccr"
        ))


def test_candidate_values_cannot_change_comparator_decisions(
    complete_selection_fixture,
) -> None:
    first = build_comparator_selection_receipt(**complete_selection_fixture)
    mutated = mutate_only_maskimpute_values(complete_selection_fixture)
    second = build_comparator_selection_receipt(**mutated)
    assert {
        method: row["selected_configuration_id"]
        for method, row in first["methods"].items()
    } == {
        method: row["selected_configuration_id"]
        for method, row in second["methods"].items()
    }
    assert first["checkpoint_file_sha256"] != second["checkpoint_file_sha256"]


def test_comparator_receipt_publication_is_create_only_and_idempotent(
    complete_checkpoint_tree: Path,
) -> None:
    repository_copy = complete_checkpoint_tree
    first = publish_comparator_selection(repository_copy)
    second = publish_comparator_selection(repository_copy)
    assert first == second
    path = repository_copy / COMPARATOR_SELECTION_RELATIVE_PATH
    payload = json.loads(path.read_text())
    payload["readiness"]["status"] = "blocked"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(ComparatorTuningError, match="existing comparator selection differs"):
        publish_comparator_selection(repository_copy)
```

Define these fixtures/helpers in the same file. `complete_selection_fixture`
uses the Task 10 record factory for all 34 configurations, adds exact completed
observed/capacity rows, and supplies a canonical synthetic 2,896-row checkpoint
whose candidate metric values are a separate mapping. `mutate_only_maskimpute_values`
deep-copies that mapping, changes only candidate metric bytes, and recomputes
the checkpoint raw/payload hashes. `selection_fixture_with_intrinsic_unavailable`
replaces every configuration cell of the named method with one declared
intrinsic terminal status/reason while preserving all 48 cells/configuration.
`complete_checkpoint_tree` writes that canonical fixture into a repository made
by the existing `_ready_repository(tmp_path)` pattern; `repository_copy` is the
returned path, not an undeclared pytest fixture. Assert the factory has exactly
2,896 unique run IDs before any receipt test.

- [ ] **Step 2: Run receipt tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'readiness or candidate_values or receipt_publication' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because receipt/readiness/publication APIs are absent.

- [ ] **Step 3: Define readiness and nonexecution identity exactly**

```python
def _nonexecution_identity(
    *,
    method_id: str,
    registry_method_sha256: str,
    authority: ComparatorTuningAuthority,
    configuration_terminal_denominator: Sequence[Mapping[str, object]],
) -> str:
    return canonical_sha256({
        "schema": "maskimpute-comparator-nonexecution-identity-v1",
        "method_id": method_id,
        "registry_method_sha256": registry_method_sha256,
        "tuning_authority_file_sha256": authority.file_sha256,
        "tuning_authority_payload_sha256": authority.payload_sha256,
        "selection_receipt_namespace": "maskimpute-comparator-selection-v1",
        "configuration_terminal_denominator": list(configuration_terminal_denominator),
    })


def _readiness(
    authority: ComparatorTuningAuthority,
    control_statuses: Mapping[str, str],
    selectable_ids: set[str],
    blocking_status_count: int,
) -> dict[str, object]:
    blockers: list[str] = []
    if any(control_statuses.get(item) != "completed" for item in authority.required_control_ids):
        blockers.append("required_control_incomplete")
    if not set(authority.established_comparator_ids) <= selectable_ids:
        blockers.append("established_comparator_unselectable")
    modern = tuple(item for item in authority.modern_core_ids if item in selectable_ids)
    if len(modern) < 3:
        blockers.append("fewer_than_three_modern_core_selectable")
    if blocking_status_count:
        blockers.append("nonscientific_incomplete_outcome_present")
    return {
        "status": "ready" if not blockers else "blocked",
        "blocker_codes": blockers,
        "required_controls_complete": not any(
            control_statuses.get(item) != "completed" for item in authority.required_control_ids
        ),
        "established_selectable_ids": [
            item for item in authority.established_comparator_ids if item in selectable_ids
        ],
        "modern_selectable_ids": list(modern),
        "modern_selectable_count": len(modern),
        "ready_comparison_population_ids": [
            item for item in authority.scheduled_same_input_ids
            if item in authority.required_control_ids or item in selectable_ids
        ],
    }
```

- [ ] **Step 4: Build the self-hashing complete receipt**

`build_comparator_selection_receipt` must first reconstruct and validate the exact production plan/checkpoint; require schema, plan hash, 2,896 unique records, exact checkpoint file/payload/input hashes, exact budget replay, no duplicate identities, and exact status denominators. It projects comparator rows only into Task 10. Controls are assessed separately. Candidate rows contribute only to the full checkpoint binding hashes.

Return this exact top-level schema:

```python
unsigned = {
    "schema_version": 1,
    "artifact_type": "maskimpute-comparator-selection-v1",
    "data_scope": "development_only",
    "final_data_used": False,
    "tuning_authority_path": "study/comparator_tuning.json",
    "tuning_authority_file_sha256": authority.file_sha256,
    "tuning_authority_payload_sha256": authority.payload_sha256,
    "selection_contract_file_sha256": selection_contract_file_sha256,
    "method_registry_file_sha256": method_registry_file_sha256,
    "checkpoint_path": "artifacts/study/development/competition-reconstruction/checkpoint.json",
    "checkpoint_file_sha256": checkpoint_file_sha256,
    "checkpoint_payload_sha256": report.checkpoint_sha256,
    "plan_sha256": report.plan_sha256,
    "input_hashes": dict(report.input_hashes),
    "dataset_ids": list(dataset_ids),
    "model_seeds": [42, 43, 44],
    "selection_metrics": list(SELECTION_METRICS),
    "methods": method_receipts_in_authority_order,
    "controls": control_receipts_in_authority_order,
    "scheduled_same_input_ids": list(authority.scheduled_same_input_ids),
    "required_control_ids": list(authority.required_control_ids),
    "established_comparator_ids": list(authority.established_comparator_ids),
    "modern_core_ids": list(authority.modern_core_ids),
    "readiness": readiness,
}
return {**unsigned, "payload_sha256": canonical_sha256(unsigned)}
```

Use these closed per-method and per-configuration key sets; later tasks consume
these names verbatim:

```python
METHOD_RECEIPT_KEYS = {
    "method_id", "selection_status", "configuration_order",
    "terminal_status_counts", "reason_histogram", "configurations",
    "pareto_configuration_ids", "selected_configuration_id",
    "selected_payload", "selected_payload_sha256",
    "registry_method_sha256", "source_authority_sha256",
    "runtime_lock_sha256", "environment_registry_sha256",
    "selected_method_identity_sha256", "nonexecution_identity_sha256",
}
CONFIGURATION_RECEIPT_KEYS = {
    "configuration_id", "configuration_payload_sha256",
    "configuration_method_identity_sha256", "is_upstream_default",
    "terminal_status_counts", "reason_histogram", "eligible",
    "eligibility_reason", "unit_ids", "unit_values", "unit_counts",
    "metric_medians", "pareto_member", "metric_rank_quarters",
    "selection_tuple",
}
```

`selection_status` is exactly `"selected"` or
`"intrinsic_terminal_no_eligible_configuration"`. Method-level counts are the
sum of its configuration counts and configurations remain in authority order.
For `"selected"`, every selected field and all six identity-body component
hashes are nonnull and the nonexecution field is null. For intrinsic-terminal
unavailability, all selected payload/identity fields are null and the
nonexecution field is nonnull. The closed loader recomputes both aggregation
levels and rejects extra, missing, reordered, or type-coerced values.

Each method receipt contains configuration order; all 48-cell status counts/reason histograms and identities for every configuration; eligibility, collapsed values, medians, Pareto membership, quarter-ranks, and tuple; selected full payload/hash and repeated preselection identity; or null selected fields plus the nonexecution identity. Raise if readiness is blocked; a private pure builder option may return blocked fixtures only for testing, but the publisher has no override.

- [ ] **Step 5: Implement secure fixed-path create-only publication and validation**

Implement module-local `_secure_read_regular` and `_immutable_publish` helpers
with the repository's existing owned-regular-file/no-follow/exclusive-hard-link
semantics. Do not import the private helper from `selection_promotion.py`; that
would create a `selection -> comparator_tuning -> selection_promotion ->
selection` cycle in Task 12. `publish_comparator_selection(repository)` accepts
no paths, methods, metrics, or thresholds. It reconstructs the production
checkpoint, builds canonical bytes plus newline, publishes create-only,
accepts concurrent byte-identical publication, rereads and semantically
validates, and rejects existing different bytes.

`load_comparator_selection_receipt` requires the exact top-level/method/configuration key sets, recomputes every identity/selection decision from checkpoint evidence when `expected_checkpoint` is supplied, and always verifies its own payload hash, raw canonical encoding, tuning authority, registry, and selection-contract bindings.

- [ ] **Step 6: Add the no-override selector script**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maskimpute_benchmark.comparator_tuning import (  # noqa: E402
    ComparatorTuningError,
    publish_comparator_selection,
)


def main() -> int:
    argparse.ArgumentParser(
        description="Select one fixed development-only configuration per comparator."
    ).parse_args()
    try:
        receipt = publish_comparator_selection(ROOT)
    except (ComparatorTuningError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "path": "artifacts/study/development/evaluation/comparator_selection.json",
        "payload_sha256": receipt["payload_sha256"],
        "readiness": receipt["readiness"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run comparator selection and CLI tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS, including tamper tests for every authority, payload, identity, metric, status, and selected tuple field.

- [ ] **Step 8: Commit the comparator receipt**

```bash
git add maskimpute_benchmark/comparator_tuning.py \
  scripts/select_comparator_configurations.py tests/test_comparator_tuning.py
git commit -m "feat: publish comparator selection receipt"
```

---

### Task 12: Project only selected comparator identities into candidate selection

**Files:**

- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Modify: `maskimpute_benchmark/development_evaluation.py:1662-1866,2467-3132`
- Modify: `maskimpute_benchmark/evaluation_manifest.py:153-588,1007-1049`
- Modify: `maskimpute_benchmark/selection.py:469-638,788-917,985-1293,1492-1677,3248-3558`
- Modify: `maskimpute_benchmark/selection_promotion.py`
- Modify: `scripts/build_development_selection_input.py`
- Modify: `tests/test_development_evaluation.py`
- Modify: `tests/test_selection_authority.py`
- Modify: `tests/test_candidate_selection.py`

**Interfaces:**

- Consumes: validated `comparator_selection.json`, exact base checkpoint, and static selection authority.
- Produces: `SelectedComparatorConfiguration`, `ComparatorSelectionProjection`, `comparator_selection_projection(receipt)`, receipt-bound schema-2/schema-4 selection inputs, and a dynamic ready `SelectionAuthority` declaration/binding population.

- [ ] **Step 1: Add failing projection and gate-population tests**

```python
def test_reconstruction_projection_keeps_only_selected_comparator_configs(
    reconstruction_evidence,
    prepared_datasets,
    comparator_receipt,
    comparator_receipt_file_sha256,
    checkpoint_directory,
    selection_authority,
) -> None:
    projection = comparator_selection_projection(
        comparator_receipt,
        receipt_file_sha256=comparator_receipt_file_sha256,
    )
    bundle = build_reconstruction_selection_records(
        reconstruction_evidence,
        checkpoint_directory=checkpoint_directory,
        prepared_datasets=prepared_datasets,
        declarations=selection_authority.declarations,
        method_bindings=selection_authority.method_bindings,
        comparator_projection=projection,
    )
    magic_records = [row for row in bundle.records if row["method"] == "magic"]
    assert magic_records
    assert {
        row["method_sha256"] for row in magic_records
    } == {projection.selected_by_method["magic"].selected_method_identity_sha256}
    selected_id = projection.selected_by_method["magic"].configuration_id
    assert all(row["configuration_id"] == selected_id for row in magic_records)
    assert not any(
        row["configuration_id"] != selected_id
        for row in magic_records
    )


def test_candidate_gate_population_is_exact_ready_projection(
    selection_payload, comparator_receipt
) -> None:
    report = select_development_candidate(selection_payload)
    expected = tuple(
        comparator_receipt["readiness"]["ready_comparison_population_ids"]
    )
    assert report.comparison_population_ids == expected
    assert "biaeimpute" in expected
    assert not any(
        configuration_id in report.comparison_population_ids
        for configuration_id in ("magic-t01", "magic-t05", "magic-t07")
    )
```

Build these tests by extending existing file-local factories, not global
fixtures. In `test_development_evaluation.py`, the checkpoint factory returns
`(checkpoint_directory, ReconstructionEvidence, prepared_datasets)`; derive
`comparator_receipt` plus its raw hash from the Task 11 synthetic checkpoint
factory, and load `selection_authority` with `require_clean=False`. In
`test_candidate_selection.py`, extend its canonical selection-payload factory
with the exact comparator binding object from Step 5. Each factory asserts the
receipt raw hash, checkpoint path, 2,896-row plan hash, and ready population
before invoking production code.

- [ ] **Step 2: Run projection tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_development_evaluation.py tests/test_candidate_selection.py \
  -k 'selected_comparator_configs or exact_ready_projection' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because every comparator configuration currently collapses to its method ID and the receipt is not consumed.

- [ ] **Step 3: Define the immutable selected projection**

Add to `comparator_tuning.py`:

```python
@dataclass(frozen=True, slots=True)
class SelectedComparatorConfiguration:
    method_id: str
    configuration_id: str
    payload: Mapping[str, object]
    payload_sha256: str
    registry_method_sha256: str
    source_authority_sha256: str
    runtime_lock_sha256: str
    environment_registry_sha256: str
    selected_method_identity_sha256: str


@dataclass(frozen=True, slots=True)
class ComparatorSelectionProjection:
    receipt_file_sha256: str
    receipt_payload_sha256: str
    selected_by_method: Mapping[str, SelectedComparatorConfiguration]
    nonexecution_identity_by_method: Mapping[str, str]
    scheduled_same_input_ids: tuple[str, ...]
    ready_comparison_population_ids: tuple[str, ...]
    scheduled_statuses_sha256: str


def comparator_selection_projection(
    receipt: Mapping[str, object],
    *,
    receipt_file_sha256: str,
) -> ComparatorSelectionProjection:
    if type(receipt) is not dict or not _SHA256.fullmatch(receipt_file_sha256):
        raise ComparatorTuningError("comparator selection projection input is invalid")
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    if receipt.get("payload_sha256") != canonical_sha256(unsigned):
        raise ComparatorTuningError("comparator selection payload checksum differs")
    methods = receipt.get("methods")
    readiness = receipt.get("readiness")
    if not isinstance(methods, Mapping) or not isinstance(readiness, Mapping):
        raise ComparatorTuningError("comparator selection projection is invalid")
    selected: dict[str, SelectedComparatorConfiguration] = {}
    unavailable: dict[str, str] = {}
    statuses = []
    for method_id, row in methods.items():
        if not isinstance(method_id, str) or not isinstance(row, Mapping):
            raise ComparatorTuningError("comparator method projection is invalid")
        statuses.append({
            "method_id": method_id,
            "status": row.get("selection_status"),
            "terminal_status_counts": row.get("terminal_status_counts"),
            "reason_histogram": row.get("reason_histogram"),
            "configuration_order": row.get("configuration_order"),
            "configurations": [
                {
                    "configuration_id": item.get("configuration_id"),
                    "configuration_payload_sha256": item.get(
                        "configuration_payload_sha256"
                    ),
                    "configuration_method_identity_sha256": item.get(
                        "configuration_method_identity_sha256"
                    ),
                    "terminal_status_counts": item.get("terminal_status_counts"),
                    "reason_histogram": item.get("reason_histogram"),
                }
                for item in row.get("configurations", [])
            ],
        })
        if row.get("selected_configuration_id") is None:
            identity = row.get("nonexecution_identity_sha256")
            if not isinstance(identity, str) or _SHA256.fullmatch(identity) is None:
                raise ComparatorTuningError("comparator nonexecution identity is invalid")
            unavailable[method_id] = identity
            continue
        configuration_id = row.get("selected_configuration_id")
        payload_sha256 = row.get("selected_payload_sha256")
        selected_identity = row.get("selected_method_identity_sha256")
        component_hashes = tuple(
            row.get(name)
            for name in (
                "registry_method_sha256", "source_authority_sha256",
                "runtime_lock_sha256", "environment_registry_sha256",
            )
        )
        if (
            not isinstance(configuration_id, str)
            or not isinstance(payload_sha256, str)
            or _SHA256.fullmatch(payload_sha256) is None
            or not isinstance(selected_identity, str)
            or _SHA256.fullmatch(selected_identity) is None
            or type(row.get("selected_payload")) is not dict
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in component_hashes
            )
        ):
            raise ComparatorTuningError("selected comparator projection is invalid")
        selected[method_id] = SelectedComparatorConfiguration(
            method_id=method_id,
            configuration_id=configuration_id,
            payload=_deep_freeze_json_mapping(row["selected_payload"]),
            payload_sha256=payload_sha256,
            registry_method_sha256=component_hashes[0],
            source_authority_sha256=component_hashes[1],
            runtime_lock_sha256=component_hashes[2],
            environment_registry_sha256=component_hashes[3],
            selected_method_identity_sha256=selected_identity,
        )
    return ComparatorSelectionProjection(
        receipt_file_sha256=receipt_file_sha256,
        receipt_payload_sha256=receipt["payload_sha256"],
        selected_by_method=MappingProxyType(selected),
        nonexecution_identity_by_method=MappingProxyType(unavailable),
        scheduled_same_input_ids=tuple(receipt["scheduled_same_input_ids"]),
        ready_comparison_population_ids=tuple(readiness["ready_comparison_population_ids"]),
        scheduled_statuses_sha256=canonical_sha256(statuses),
    )
```

Implement `_deep_freeze_json_mapping` beside the authority's existing recursive
JSON freezer: accept only exact JSON primitives, finite floats, lists, and
string-key dictionaries; recursively convert lists to tuples and mappings to
`MappingProxyType`. Before the loop, require the exact Task 11 key sets, method
order equal the receipt's scheduled comparator order, readiness status
`"ready"`, exact selected/nonexecution disjointness, and raw/payload SHA types.
The scheduled-status hash above intentionally covers complete per-configuration
counts, reasons, payload hashes, and preselection identities.

- [ ] **Step 4: Make reconstruction mapping selected-config aware**

Change the exact interface to:

```python
def reconstruction_selection_method(
    run: Mapping[str, object],
    declared: set[str],
    comparator_projection: ComparatorSelectionProjection,
) -> str | None:
    configuration_id = run.get("configuration_id")
    if (
        run.get("configuration_kind") == "candidate_search"
        and isinstance(configuration_id, str)
        and configuration_id in declared
    ):
        return configuration_id
    method_id = run.get("method_id")
    if not isinstance(method_id, str) or method_id not in declared:
        return None
    selected = comparator_projection.selected_by_method.get(method_id)
    if selected is None:
        return method_id if method_id in {"observed", "capacity-matched-ae"} else None
    if (
        configuration_id != selected.configuration_id
        or run.get("configuration_payload_sha256") != selected.payload_sha256
        or run.get("configuration_method_identity_sha256")
        != selected.selected_method_identity_sha256
    ):
        return None
    return method_id
```

Pass the projection through `build_reconstruction_selection_records`. For selected comparators set selection record `method_sha256` to `selected_method_identity_sha256` and include `configuration_id`, `configuration_payload_sha256`, and `configuration_method_identity_sha256` in each audit row. Nonselected configurations remain only in the comparator receipt/checkpoint.

- [ ] **Step 5: Bind the receipt into selection inputs and manifests**

Before loading reconstruction evidence in `build_development_selection_input`, fixed-load and semantically validate `comparator_selection.json`. Add this exact object to schema-2 source, promoted schema-4 input, and evaluation manifest:

```python
"comparator_selection": {
    "path": COMPARATOR_SELECTION_RELATIVE_PATH,
    "file_sha256": projection.receipt_file_sha256,
    "payload_sha256": projection.receipt_payload_sha256,
    "scheduled_statuses_sha256": projection.scheduled_statuses_sha256,
    "ready_comparison_population_ids": list(projection.ready_comparison_population_ids),
}
```

Update every closed field set and raw artifact inventory in `development_evaluation.py` and `evaluation_manifest.py`. Rebuild the production plan with tuning/runtime identity fields before accepting the checkpoint. Reread the comparator receipt after writing the selection input and require byte/hash equality.

- [ ] **Step 6: Derive dynamic selection declarations and bindings**

Add to `selection.py`:

```python
def _authority_with_comparator_projection(
    authority: SelectionAuthority,
    projection: ComparatorSelectionProjection,
) -> SelectionAuthority:
    population = set(projection.ready_comparison_population_ids)
    declarations = tuple(
        replace(
            row,
            required_for_claim=(row.role == "candidate" or row.id in population),
        )
        for row in authority.declarations
        if row.role == "candidate" or row.id in population
    )
    retained_ids = {row.id for row in declarations}
    bindings = {
        method_id: digest
        for method_id, digest in authority.method_bindings.items()
        if method_id in retained_ids
        and method_id not in projection.scheduled_same_input_ids
    }
    for method_id in authority.required_control_ids:
        bindings[method_id] = authority.method_bindings[method_id]
    for method_id, selected in projection.selected_by_method.items():
        bindings[method_id] = selected.selected_method_identity_sha256
    return replace(
        authority,
        declarations=declarations,
        method_bindings=MappingProxyType(bindings),
    )
```

In `_select_for_repository`, load the fixed receipt, compare the input's binding object exactly, derive this authority, and include `comparison_population_ids`, comparator receipt file/payload hashes, and scheduled-status hash in `SelectionReport`. `_rank_summary`, `_pareto_dominators`, and `_assessment` must operate only on this closed declared population and fail on a missing required record rather than silently skipping it.

Unavailable methods have no `method_bindings` entry; their identity remains
only in `projection.nonexecution_identity_by_method` and scheduled-status
evidence. Add the three comparator audit fields to `_RECORD_FIELDS`,
`_ValidatedRecord`, development manifest validation, selection promotion
schemas 2/3/4, revision validators, and result-hash projections. Reject stale
registry-default bindings rather than retaining them.

- [ ] **Step 7: Run development, manifest, promotion, and selection tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_development_evaluation.py tests/test_selection_authority.py \
  tests/test_candidate_selection.py tests/test_selection_promotion.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS; changing a nonselected comparator record changes receipt/checkpoint bindings but never creates a candidate-comparison identity.

- [ ] **Step 8: Commit selected-only projection**

```bash
git add maskimpute_benchmark/comparator_tuning.py \
  maskimpute_benchmark/development_evaluation.py \
  maskimpute_benchmark/evaluation_manifest.py maskimpute_benchmark/selection.py \
  maskimpute_benchmark/selection_promotion.py \
  scripts/build_development_selection_input.py tests/test_development_evaluation.py \
  tests/test_selection_authority.py tests/test_candidate_selection.py \
  tests/test_selection_promotion.py
git commit -m "feat: project selected comparator identities"
```

---

### Task 13: Make v28/v29 revision execution candidate-only and reuse base comparators

**Files:**

- Modify: `maskimpute_benchmark/runner.py:973-1337,1473-1624,6655-6719`
- Modify: `maskimpute_benchmark/revisions.py:420-600`
- Modify: `maskimpute_benchmark/revision_evaluation.py:164-249,685-832,929-1169`
- Modify: `tests/test_revision_authority.py`
- Modify: `tests/test_revision_evaluation.py`
- Modify: `tests/test_benchmark_runner.py`

**Interfaces:**

- Consumes: base comparator receipt/projection and one activated v28 or v29 MaskImpute configuration.
- Produces: `RunnerAuthority.plan_scope` equal to `base_full_panel` or `revision_candidate_only`, exact 48-row revision plans, and revision selection sources that bind/reuse base selected comparator/control rows.

- [ ] **Step 1: Add failing candidate-only revision tests**

```python
@pytest.mark.parametrize(
    "loader",
    (load_v28_revision_authority, load_v29_revision_authority),
)
def test_revision_runner_authority_is_candidate_only(loader) -> None:
    authority = loader()
    assert authority.plan_scope == "revision_candidate_only"
    assert len(authority.configurations) == 1
    assert authority.configurations[0].method_id == "maskimpute"


def test_revision_plan_contains_exactly_one_48_row_maskimpute_candidate(
    activated_v28_authority, registry, development_bindings
) -> None:
    plan = build_competition_plan(
        registry,
        development_bindings,
        activated_v28_authority,
        execution_environment_sha256="2" * 64,
        runtime_lock_sha256="1" * 64,
    )
    assert len(plan.entries) == 48
    assert {entry.method_id for entry in plan.entries} == {"maskimpute"}
    assert {entry.configuration_id for entry in plan.entries} == {
        "v28-c01-nb-parent-c03"
    }
```

Use existing revision test repository builders to activate v28/v29 rather than
inventing session fixtures. `activated_v28_authority` is the return value of
the v28 loader against that prepared repository; `registry` loads its copied
`study/methods.json`; and `development_bindings` is
`validate_development_manifest_payload(_manifest_payload())`. Update the
parametrized loader wrapper to pass the repository/activation arguments each
real loader currently requires. Assert the tracked IDs are exactly
`v28-c01-nb-parent-c03` and `v29-c01-structure-parent-v28-c01` in their
respective tests.

- [ ] **Step 2: Run revision tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_revision_authority.py tests/test_revision_evaluation.py \
  tests/test_benchmark_runner.py -k 'candidate_only or exact_48_candidate_rows' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because revision authorities still include capacity control and planning adds every registry comparator.

- [ ] **Step 3: Complete and enforce the closed plan scope introduced in Task 5**

Task 5 has already added this exact field to `RunnerAuthority`:

```python
    plan_scope: Literal["base_full_panel", "revision_candidate_only"] = "base_full_panel"
```

Base loading uses `base_full_panel`. v28/v29 loaders return only `(candidate,)`, set `revision_candidate_only`, and bind the base comparator-selection file/payload hashes into their authority bodies. Remove the capacity-control field from revision authority bodies.

Branch `RunnerAuthority.__post_init__`: base scope requires the exact capacity
control and base counts; revision scope requires exactly one MaskImpute
candidate and zero capacity/comparator/registry rows. When
`assemble_revision_evaluation` rebuilds the revision plan, read and validate
both `execution_environment_sha256` and `runtime_lock_sha256` from checkpoint
`input_hashes` and pass both to `build_competition_plan`.

In `build_competition_plan`, derive `planned_specs` as:

```python
    planned_specs = (
        tuple(spec for spec in registry.methods if spec.id == "maskimpute")
        if authority.plan_scope == "revision_candidate_only"
        else tuple(
            spec for spec in registry.methods
            if spec.execution_scope == "same_input_required"
        )
    )
```

For revision scope require one candidate configuration, 48 rows, and zero `observed`, capacity, registry, or comparator-tuning rows. For base scope retain the exact Task 5 counts.

- [ ] **Step 4: Preserve comparator receipt fields through revision authority**

Replace `derive_extended_selection_authority`'s old comparator field copy with all four sets plus:

```python
        comparator_tuning_file_sha256=base.comparator_tuning_file_sha256,
        comparator_tuning_payload_sha256=base.comparator_tuning_payload_sha256,
```

Revision activation validation requires the base input/report comparator-selection file hash, payload hash, scheduled-status hash, and ready population to match exactly. Any revision source that omits or changes the binding fails before execution.

Before `derive_extended_selection_authority`, fixed-load and validate the base
comparator receipt and call `_authority_with_comparator_projection`; extending
the static authority directly is forbidden. Add explicit revision authority
fields for comparator-selection path, file hash, payload hash, scheduled-status
hash, and ready population, and repeat them in activation/evaluation manifests.

- [ ] **Step 5: Reuse base controls and selected comparator rows byte-for-byte**

In `assemble_revision_evaluation`, validate the revision checkpoint contains only the selected revision candidate's 48 rows. Keep `combine_selection_rows(base_records, revision_records, ...)` but add exact assertions:

```python
    inherited = tuple(
        row for row in base_records
        if row.get("method") not in {attempt.configuration_id for attempt in authority.attempts}
    )
    combined_inherited = tuple(
        row for row in combined_records
        if row.get("method") not in {attempt.configuration_id for attempt in authority.attempts}
    )
    if combined_inherited != inherited:
        raise RevisionEvaluationError("revision comparator/control rows differ from base evidence")
```

Bind `base_comparator_selection` into revision evaluation manifests and source projections. Do not copy comparator raw matrices into the revision checkpoint; references remain bound to base evidence.

- [ ] **Step 6: Run revision, activation, and freeze-stage tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_revision_authority.py tests/test_revision_evaluation.py \
  tests/test_selection_promotion.py tests/test_freeze_publication_round.py \
  -k 'revision or activation or base_comparator' \
  -q -W error -p no:cacheprovider
```

Expected: PASS with 48-row candidate-only revision checkpoints and unchanged base comparator receipt hashes.

- [ ] **Step 7: Commit candidate-only revisions**

```bash
git add maskimpute_benchmark/runner.py maskimpute_benchmark/revisions.py \
  maskimpute_benchmark/revision_evaluation.py tests/test_revision_authority.py \
  tests/test_revision_evaluation.py tests/test_benchmark_runner.py \
  tests/test_selection_promotion.py tests/test_freeze_publication_round.py
git commit -m "fix: reuse base comparators in revisions"
```

---

### Task 14: Freeze selected payloads and unavailable nonexecution identities

**Files:**

- Modify: `maskimpute_benchmark/publication_freeze.py:78-103,2082-2286,2551-3097,3195-3614`
- Modify: `tests/test_freeze_publication_round.py`

**Interfaces:**

- Consumes: validated comparator tuning authority, comparator selection receipt/projection, active candidate stage, and base execution evidence.
- Produces: a frozen payload with exact four comparator sets, complete scheduled status denominator, selected full payload/id/hash/identity map, unavailable nonexecution map, and receipt/authority hashes.

- [ ] **Step 1: Add failing freeze payload and tamper tests**

```python
def test_frozen_payload_binds_selected_comparator_payloads_and_status_denominator(
    frozen_payload_fixture,
) -> None:
    payload = frozen_payload_fixture()
    assert payload["scheduled_same_input_ids"] == [
        "observed", "capacity-matched-ae", "alra", "magic", "dca", "scvi",
        "saver", "scziva", "afmf", "biaeimpute", "sccr", "scsdae",
    ]
    assert payload["required_control_ids"] == ["observed", "capacity-matched-ae"]
    assert payload["established_comparator_ids"] == ["alra", "magic", "dca", "scvi", "saver"]
    assert payload["modern_core_ids"] == ["scziva", "afmf", "biaeimpute", "sccr"]
    selected = payload["selected_comparator_configurations"]
    assert selected["magic"]["configuration_id"] != "registry-default"
    assert canonical_sha256(selected["magic"]["payload"]) == selected["magic"]["payload_sha256"]
    assert selected["magic"]["selected_method_identity_sha256"]
    assert tuple(
        row["method_id"] for row in payload["scheduled_same_input_statuses"]
    ) == tuple(payload["scheduled_same_input_ids"])
    assert len(payload["scheduled_same_input_statuses"]) == 12


def test_frozen_payload_rejects_comparator_receipt_or_selected_payload_tamper(
    repository_copy, prepared_frozen_method
) -> None:
    receipt_path = repository_copy / COMPARATOR_SELECTION_RELATIVE_PATH
    receipt = json.loads(receipt_path.read_text())
    receipt["methods"]["magic"]["selected_payload"]["diffusion_time"] = 99
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(PublicationFreezeError, match="comparator selection"):
        validate_frozen_method(repository_copy)
```

Extend the existing freeze test repository/receipt factories rather than adding
undeclared fixtures. `frozen_payload_fixture` wraps the current
`build_frozen_method_payload` helper with the Task 11 comparator receipt;
`prepared_frozen_method` is the existing prepared round directory after both
new artifacts are added; and `repository_copy` is the path returned by that
factory. Each factory validates all raw hashes before the test mutates one
file. Add a parametrized component-hash tamper test over the four new component
fields plus tuning file/payload hashes and selected identity.

- [ ] **Step 2: Run freeze tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -k 'selected_comparator_payloads or comparator_receipt' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because freeze still consumes `required_comparator_ids` and reconstructs comparator defaults later.

- [ ] **Step 3: Add both comparator artifacts to the closed inventory**

Add:

```python
_COMMON_TRACKED_PATHS["comparator_tuning"] = "study/comparator_tuning.json"
_COMMON_DEVELOPMENT_PATHS["comparator_selection"] = (
    "artifacts/study/development/evaluation/comparator_selection.json"
)
```

Ensure tree/stage receipts, artifact key equality, clean Git validation, operational evidence validation, and round freeze all bind these paths. The generated selection receipt is a development artifact, not a tracked Git authority.

- [ ] **Step 4: Replace the frozen builder comparator interface**

Change `build_frozen_method_payload` parameters from `required_comparator_ids` to:

```python
    comparator_tuning_authority: ComparatorTuningAuthority,
    comparator_selection_receipt: Mapping[str, object],
    comparator_selection_file_sha256: str,
```

Validate the receipt again against the base checkpoint and exact artifact binding. Derive:

```python
    projection = comparator_selection_projection(
        comparator_selection_receipt,
        receipt_file_sha256=comparator_selection_file_sha256,
    )
    selected_comparators = {
        method_id: {
            "configuration_id": selected.configuration_id,
            "payload": dict(selected.payload),
            "payload_sha256": selected.payload_sha256,
            "tuning_authority_file_sha256": comparator_tuning_authority.file_sha256,
            "tuning_authority_payload_sha256": comparator_tuning_authority.payload_sha256,
            "comparator_selection_file_sha256": comparator_selection_file_sha256,
            "comparator_selection_payload_sha256": projection.receipt_payload_sha256,
            "registry_method_sha256": selected.registry_method_sha256,
            "source_authority_sha256": selected.source_authority_sha256,
            "runtime_lock_sha256": selected.runtime_lock_sha256,
            "environment_registry_sha256": selected.environment_registry_sha256,
            "selected_method_identity_sha256": selected.selected_method_identity_sha256,
        }
        for method_id, selected in projection.selected_by_method.items()
    }
```

Extend `SelectedComparatorConfiguration` in Task 12 with the four identity-body
components shown above; its tuning hashes are already present in the authority
and are repeated in every frozen selected row. Recompute the domain-separated
identity from all seven components before freezing. Add one tamper regression
for each component.

Add top-level arrays for all four sets, `ready_comparison_population_ids`,
`selected_comparator_configurations`,
`unavailable_comparator_nonexecution_identities`, and one ordered
`scheduled_same_input_statuses` table whose 12 rows are the exact union of the
two control receipts and ten comparator receipts. Every row contains method ID,
aggregate status, terminal counts, and reason histogram. Add a
`comparator_selection_binding` object with path/file/payload/scheduled-status
hashes. Remove `required_comparator_ids` from new payloads.

- [ ] **Step 5: Make method-panel availability receipt-authoritative**

`_method_panel` must use comparator receipt method status rather than infer a selected configuration from aggregate method records. Established unavailability blocks before this function. Modern/scSDAE intrinsic unavailability is accepted only if every planned cell is completed or intrinsic terminal and the receipt provides the same nonexecution identity. Any blocked/budget/infrastructure status fails freeze.

For a selected comparator add `selected_comparator_configuration` to its method row. For an unavailable comparator set it to `None`, set `nonexecution_identity_sha256`, use disposition `explicit_reason_coded_unavailable`, and retain the complete status/reason denominator. Controls/candidate keep their existing configuration authority.

- [ ] **Step 6: Rebuild and revalidate from raw authorities**

In `_expected_frozen_method` and `_validate_clean_frozen_method`, secure-read the tuning authority and selection receipt, validate raw and payload hashes, reconstruct projection/selection decisions from the checkpoint, and pass them to the new builder. Reread both artifacts after building and require exact bytes/hash equality. Any coherent frozen-method rehash after receipt tamper must still fail.

- [ ] **Step 7: Run freeze and stage-inventory tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS, including base/v28/v29 stage reuse, intrinsic modern unavailability, and all comparator receipt/payload/identity tamper cases.

- [ ] **Step 8: Commit frozen comparator authority**

```bash
git add maskimpute_benchmark/publication_freeze.py \
  tests/test_freeze_publication_round.py
git commit -m "feat: freeze selected comparator payloads"
```

---

### Task 15: Preserve exact final and trajectory denominators without defaults

**Files:**

- Modify: `maskimpute_benchmark/final_runner.py:1272-1315,3963-4407`
- Modify: `maskimpute_benchmark/runner.py:635-747,1424-1447`
- Modify: `tests/test_final_runner.py`

**Interfaces:**

- Consumes: frozen selected comparator map or frozen nonexecution identity map.
- Produces: `FrozenPlanMethodAuthority`, exact selected/nonexecution `AuthorizedConfiguration` rows, a 1,760-row final plan, and a 44-row trajectory plan.

- [ ] **Step 1: Add failing exact cardinality and payload tests**

```python
def test_final_plan_is_1760_and_all_selectable_split_is_1480_280(
    frozen_method, registry, final_bindings
) -> None:
    plan = build_final_execution_plan(
        frozen_method, registry, final_bindings,
        execution_claim_sha256="1" * 64,
        execution_environment_sha256="2" * 64,
        execution_authority_sha256="3" * 64,
    )
    assert len(plan.entries) == 1_760
    assert sum(entry.action == "execute" for entry in plan.entries) == 1_480
    assert sum(entry.action == "not_applicable" for entry in plan.entries) == 280
    magic = next(value for value in plan.configurations if value.method_id == "magic")
    assert magic.kind == "comparator_tuning"
    assert magic.configuration_id == frozen_method["selected_comparator_configurations"]["magic"]["configuration_id"]


def test_unavailable_stochastic_comparator_reclassifies_120_seeded_rows(
    frozen_method_with_unavailable_biae, registry, final_bindings
) -> None:
    plan = build_final_execution_plan(
        frozen_method_with_unavailable_biae, registry, final_bindings,
        execution_claim_sha256="1" * 64,
        execution_environment_sha256="2" * 64,
        execution_authority_sha256="3" * 64,
    )
    rows = [entry for entry in plan.entries if entry.run.method_id == "biaeimpute"]
    assert len(plan.entries) == 1_760
    assert len(rows) == 120
    assert {entry.run.model_seed for entry in rows} == {42, 43, 44}
    assert all(entry.action == "not_applicable" for entry in rows)
    assert {entry.run.nonexecution_identity_sha256 for entry in rows} == {
        frozen_method_with_unavailable_biae[
            "unavailable_comparator_nonexecution_identities"
        ]["biaeimpute"]
    }


def test_trajectory_always_has_44_rows_and_preserves_three_unavailable_seeds(
    trajectory_plan_with_unavailable_biae,
    frozen_method_with_unavailable_biae,
) -> None:
    plan = trajectory_plan_with_unavailable_biae
    rows = [entry for entry in plan.entries if entry.run.method_id == "biaeimpute"]
    assert len(plan.entries) == 44
    assert len(rows) == 3
    assert {entry.run.model_seed for entry in rows} == {42, 43, 44}
    assert all(entry.action == "not_applicable" for entry in rows)
    assert len({entry.reason for entry in rows}) == 1
    assert {entry.run.nonexecution_identity_sha256 for entry in rows} == {
        frozen_method_with_unavailable_biae[
            "unavailable_comparator_nonexecution_identities"
        ]["biaeimpute"]
    }
    assert all(
        entry.run.configuration_kind == "comparator_nonexecution" for entry in rows
    )
```

Extend the existing `tests/test_final_runner.py` `_registry`, `_bindings`, and
frozen-method builders. The full fixtures use the tracked 44-row method
registry/configuration denominator; the unavailable fixture moves only
BiAEImpute from the selected map to the nonexecution map while retaining its
status row and three seeds. `final_bindings` is the existing canonical 40-dataset
binding factory. Do not make these pytest session fixtures: keep them
file-local so subset-registry unit tests remain independent.

- [ ] **Step 2: Run final-plan tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_final_runner.py -k '1760 or 1480 or unavailable_stochastic or always_has_44' \
  -q -W error -p no:cacheprovider
```

Expected: unavailable stochastic methods currently collapse to one seed-null row and comparator configs are registry defaults.

- [ ] **Step 3: Represent exact frozen per-method authority**

Add:

```python
@dataclass(frozen=True, slots=True)
class FrozenPlanMethodAuthority:
    method_id: str
    configuration: AuthorizedConfiguration
    action: Literal["execute", "not_applicable"]
    reason: str | None
    seeds: tuple[int | None, ...]
```

For a selected comparator, `_configuration_for_method` must load its exact frozen full payload/hash/identity and construct `kind="comparator_tuning"`; decode it immediately and require exact round-trip. It must never call `registry_default` for a method present in the tuning authority.

`_frozen_method_plan_authority` returns exactly one
`FrozenPlanMethodAuthority` per registry method. For the ten tuning methods,
the selected and nonexecution maps must be disjoint and their union must equal
the tuning-authority method order. Registry fallback is permitted only for a
method outside that ten-method set. Selected comparator construction passes
all frozen registry/source/runtime/environment/tuning component hashes and
requires `configuration_method_identity_sha256` to equal the repeated selected
identity.

For an unavailable comparator construct:

```python
AuthorizedConfiguration.create(
    method_id=method_id,
    configuration_id=f"nonexecution-{method_id}",
    kind="comparator_nonexecution",
    payload={
        "schema": "maskimpute-frozen-comparator-nonexecution-v1",
        "method_id": method_id,
        "reason": row["final_applicability"]["non_run_reason"],
        "nonexecution_identity_sha256": nonexecution_identity,
    },
    requires_count_score=False,
    requires_calibration=False,
    nonexecution_identity_sha256=nonexecution_identity,
)
```

This payload is not dispatchable and contains no adapter default.

- [ ] **Step 4: Preserve seeds for unavailable scheduled stochastic methods**

Replace `_frozen_final_applicability` with a version that accepts the exact configuration authority. Its seed branch is:

```python
    if rule == "all_final_datasets":
        return "execute", None, DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
    if (
        rule == "never"
        and spec.execution_scope == "same_input_required"
        and configuration.kind == "comparator_nonexecution"
    ):
        return "not_applicable", raw_reason, (
            DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
        )
    return "not_applicable", raw_reason, (None,)
```

Set `RunPlanEntry.configuration_method_identity_sha256` for selected comparators and `nonexecution_identity_sha256` for unavailable ones. Preserve the latter across run IDs, records, execution manifests, and evaluation replay.

Define final and trajectory run-ID identity tokens explicitly: selected
comparators use `configuration_method_identity_sha256`; comparator
nonexecutions use `nonexecution_identity_sha256`; legacy kinds use
`configuration_sha256`. Update `_final_run_id`, `_trajectory_run_id`, embedded
plan replay, and tamper tests to use that token. A nonexecution row must contain
no selected adapter payload or registry default.

- [ ] **Step 5: Assert structural counts in both builders**

In `build_final_execution_plan` and `build_trajectory_execution_plan`, first
compare the supplied registry raw hash/order with the frozen canonical registry.
When that comparison identifies the production registry, enforce these counts:

```python
    if len(entries) != 1_760:
        raise FinalRunnerContractError("final structural denominator must equal 1760")
```

and:

```python
    if len(entries) != 44:
        raise FinalRunnerContractError("trajectory structural denominator must equal 44")
```

Do not apply production cardinality assertions when existing unit tests supply
an explicitly synthetic subset registry; all other frozen bindings and schema
checks still apply. For full all-selectable
fixtures assert 1,480 executable/280 nonexecution final rows. Unavailability
changes only action/reason/identity, never total or stochastic seed count.

- [ ] **Step 6: Run final, trajectory, downstream replay, and final-analysis tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_final_runner.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS with all final/trajectory embedded-plan identity fields updated.
Downstream and final-analysis schemas run in Task 16 after their exact field
sets are migrated.

- [ ] **Step 7: Commit seed-preserving final plans**

```bash
git add maskimpute_benchmark/final_runner.py maskimpute_benchmark/runner.py \
  tests/test_final_runner.py
git commit -m "fix: preserve frozen comparator denominators"
```

---

### Task 16: Propagate frozen identities through scaling, downstream analysis, and claims

**Files:**

- Modify: `maskimpute_benchmark/runner.py:2729-2793`
- Modify: `maskimpute_benchmark/scaling.py:229-265,1182-1297,3570-3685`
- Modify: `maskimpute_benchmark/downstream_evidence.py:333-340,1126-1160,1530-1600,3349-3605,4840-4860`
- Modify: `maskimpute_benchmark/final_analysis.py:1762-2160,2940-3130`
- Modify: `maskimpute_benchmark/publication_synthesis.py:180-230,1090-1140`
- Modify: `maskimpute_benchmark/evaluation_manifest.py`
- Modify: `tests/test_scaling_panel.py`
- Modify: `tests/test_downstream_evidence.py`
- Modify: `tests/test_final_analysis.py`
- Modify: `tests/test_publication_synthesis.py`
- Modify: `tests/test_selection_authority.py`
- Modify: `tests/test_benchmark_runner.py`

**Interfaces:**

- Consumes: exact frozen selected configuration/nonexecution fields and final/trajectory entries.
- Produces: scaling plans with selected MAGIC/DCA/scVI payloads, downstream method-artifact identities, analysis reports with separate scheduled/numerical denominators, and claim gates that cannot claim superiority over unavailable methods.

- [ ] **Step 1: Add failing scaling and claim-denominator tests**

```python
def test_scaling_uses_exact_frozen_comparator_payloads(scaling_authority) -> None:
    frozen = scaling_authority.frozen_method["selected_comparator_configurations"]
    planned = {row.method_id: row for row in scaling_authority.plan.configurations}
    for method_id in ("magic", "dca", "scvi"):
        assert planned[method_id].configuration_id == frozen[method_id]["configuration_id"]
        assert planned[method_id].configuration_payload_sha256 == frozen[method_id]["payload_sha256"]
        assert planned[method_id].configuration_method_identity_sha256 == frozen[method_id]["selected_method_identity_sha256"]


def test_publication_claims_separate_scheduled_and_numerical_denominators(
    synthesis_fixture_with_unavailable_biae,
) -> None:
    synthesis = _build_publication_synthesis(
        synthesis_fixture_with_unavailable_biae
    )
    assert "biaeimpute" in synthesis["scheduled_same_input_ids"]
    assert "biaeimpute" not in synthesis["numerical_comparison_population_ids"]
    assert synthesis["execution_status_by_method"]["biaeimpute"]["status"] == "unavailable"
    assert (
        synthesis["claim_permissions"]["superiority_by_method"]["biaeimpute"]
        == "unavailable_uncompared_method"
    )
```

Update `tests/test_scaling_panel.py::_configurations()` to construct exact bound
MAGIC/DCA/scVI comparator configurations from the frozen fixture rather than
registry defaults; `scaling_authority` is the existing authority/plan fixture
using that helper. Add payload, each component-identity, executor-receipt,
checkpoint, and stored-evidence tamper cases. Build
`synthesis_fixture_with_unavailable_biae` as the existing
`_LoadedPublicationEvidence` fixture with only the frozen BiAEImpute selection
changed to the validated nonexecution/status form; pass that object directly to
`_build_publication_synthesis`.

- [ ] **Step 2: Run scaling/analysis tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_scaling_panel.py tests/test_publication_synthesis.py \
  -k 'exact_frozen_comparator or scheduled_and_numerical' \
  -q -W error -p no:cacheprovider
```

Expected: FAIL because scaling reconstructs registry defaults and synthesis expects `required_comparator_ids`.

- [ ] **Step 3: Extend scaling plan identity fields and reject defaults**

Add `configuration_payload_sha256`, `configuration_method_identity_sha256`, and `nonexecution_identity_sha256` to `ScalingPlanEntry`; the five fixed scaling methods must all be executable, so nonexecution is always null. In `build_scaling_plan`, require MAGIC/DCA/scVI `kind == "comparator_tuning"`, exact selected identities, and no `registry-default`. `load_scaling_execution_authority` compares every field to the frozen selected map before materializing datasets.

Pass the same fields through `_run_plan_entry`, executor receipts, checkpoints, and scaling evidence. Rehashing a payload or identity in either frozen method or scaling checkpoint must fail.

- [ ] **Step 4: Update downstream configuration and run schemas**

Change `_CONFIGURATION_FIELDS` to include:

```python
{
    "configuration_payload_sha256",
    "configuration_method_identity_sha256",
    "nonexecution_identity_sha256",
}
```

Add the same fields to every embedded run-field set. Replace `_method_artifact_sha256` with:

```python
def _method_artifact_sha256(configuration: AuthorizedConfiguration) -> str:
    if configuration.kind == "comparator_tuning":
        if configuration.configuration_method_identity_sha256 is None:
            raise DownstreamEvidenceError("selected comparator identity is absent")
        return configuration.configuration_method_identity_sha256
    if configuration.kind == "comparator_nonexecution":
        if configuration.nonexecution_identity_sha256 is None:
            raise DownstreamEvidenceError("comparator nonexecution identity is absent")
        return configuration.nonexecution_identity_sha256
    return configuration.configuration_sha256
```

Development downstream evidence accepts only the selected comparator configuration for numerical endpoints. Final/trajectory downstream evidence retains nonexecution rows with null endpoint values, upstream terminal status, exact reason, and nonexecution identity.

First add all three fields to `runner.RawRunResult`, both constructors, and
`to_dict`; copy them from `RunPlanEntry` for completed and noncompleted runs and
compare each directly with its `AuthorizedConfiguration`. Then migrate
`downstream_evidence._RAW_RUN_FIELDS`, `_CONFIGURATION_FIELDS`,
`_SOURCE_PLAN_RUN_FIELDS`, `DownstreamPlanEntry`, every `to_dict`/decoder,
configuration lookup key, stored endpoint record, and replay comparison.
Propagate the same exact fields through scaling executor receipts/checkpoints
and final-analysis trajectory replay. A change to one field must fail at the
first closed schema boundary, not only at a later aggregate hash.

- [ ] **Step 5: Replace overloaded comparator fields in analysis/synthesis**

Final analysis and publication synthesis consume these exact frozen fields:

```python
scheduled_same_input_ids = tuple(frozen["scheduled_same_input_ids"])
numerical_comparison_population_ids = tuple(frozen["ready_comparison_population_ids"])
```

Controls remain visible but are classified separately. Every status table iterates `scheduled_same_input_ids`; numerical ranks/Pareto/effects iterate the ready population and require each selected method's applicable cells. An unavailable method receives a complete reason/status denominator and explicit `superiority_claim_status="unavailable_uncompared_method"`; no headline/caption/table may imply a win over it.

Update output field names from `required_comparator_ids` to `scheduled_same_input_ids` plus `numerical_comparison_population_ids`. Validation fails on omitted selected methods or silently deleted unavailable methods.

Run `rg -n 'required_comparator_ids' maskimpute_benchmark tests` and migrate
every live constructor/consumer in this task. Then remove Task 3's temporary
compatibility property; the only remaining occurrences may be historical design
documents or explicit negative migration tests.

Extend `_build_publication_synthesis(_LoadedPublicationEvidence)` rather than
adding a new public API. Its closed output adds
`execution_status_by_method` in scheduled order and
`claim_permissions["superiority_by_method"]`, mapping every scheduled method to
exactly one of `allowed`, `control_not_a_superiority_target`,
`unavailable_uncompared_method`, or `insufficient_completed_cells`. Keep
`generate_publication_synthesis(repository, round_dir)` as the sole public
entry point and update its exact-key validation.

Add two later-cell regressions: a selected comparator with one final timeout
and one trajectory resource failure remains in both scheduled and numerical
method sets, retains null endpoint values plus exact reasons for those cells,
and receives `insufficient_completed_cells` only for unsupported comparisons.
It must never be deleted or relabeled as development-time unavailable.

Publication-asset export remains the separate subproject named in the global
completion boundary. In this task, add a stale-descendant regression that
changes the comparator selection/frozen identity beneath a coherently rehashed
old synthesis/asset manifest and requires synthesis authorization to fail.
This closes authority-tamper coverage without generating figures or claiming
that asset export itself is complete.

- [ ] **Step 6: Run scaling, downstream, analysis, and synthesis suites**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_scaling_panel.py tests/test_downstream_evidence.py \
  tests/test_final_analysis.py tests/test_publication_synthesis.py \
  tests/test_selection_authority.py tests/test_benchmark_runner.py \
  -q -W error -p no:cacheprovider
```

Expected: PASS with exact frozen selected payloads and complete unavailable status rows.

- [ ] **Step 7: Commit downstream propagation**

```bash
git add maskimpute_benchmark/scaling.py maskimpute_benchmark/downstream_evidence.py \
  maskimpute_benchmark/final_analysis.py \
  maskimpute_benchmark/publication_synthesis.py \
  maskimpute_benchmark/evaluation_manifest.py tests/test_scaling_panel.py \
  tests/test_downstream_evidence.py tests/test_final_analysis.py \
  tests/test_publication_synthesis.py tests/test_selection_authority.py \
  maskimpute_benchmark/runner.py tests/test_benchmark_runner.py
git commit -m "feat: propagate frozen comparator identities"
```

---

### Task 17: Document the fair comparison for Genome Biology and close repository hygiene

**Files:**

- Modify: `docs/development-selection-workflow.md`
- Modify: `paper/manuscript.tex:111-131`
- Modify: `paper/references.bib`
- Modify: `paper/submission_checklist.md`
- Modify: `docs/genome-biology-submission-checklist.md`
- Modify: `tests/test_publication_repository_hygiene.py`
- Modify: `tests/test_comparator_tuning.py`

**Interfaces:**

- Consumes: final authority/receipt names and denominator semantics.
- Produces: venue-aligned Methods text, execution checklist gates, and static rejection of caches, temporary outputs, or accidental scientific artifacts.

- [ ] **Step 1: Add failing documentation and hygiene assertions**

```python
def test_manuscript_discloses_development_only_comparator_selection() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    required = (
        "thirty-four",
        "sixteen development datasets",
        "three model seeds",
        "Pareto",
        "quarter-rank",
        "BiAEImpute",
        "methods with no eligible development configuration will remain in the scheduled denominator",
    )
    assert all(fragment in manuscript for fragment in required)


def test_repository_has_no_generated_comparator_evidence_or_cache() -> None:
    forbidden = (
        ROOT / "artifacts/study/development/evaluation/comparator_smoke.json",
        ROOT / "artifacts/study/development/evaluation/comparator_selection.json",
    )
    assert not any(os.path.lexists(path) for path in forbidden)
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout.decode().split("\0")
    assert not any(
        {"__pycache__", ".pytest_cache", ".ruff_cache"} & set(Path(name).parts)
        or name.endswith((".pyc", ".tmp", ".partial"))
        for name in listed if name
    )
```

- [ ] **Step 2: Run documentation/hygiene tests red**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_publication_repository_hygiene.py tests/test_comparator_tuning.py \
  -k 'manuscript_discloses or generated_comparator_evidence' \
  -q -W error -p no:cacheprovider
```

Expected: manuscript assertion FAIL; hygiene fixture must explicitly ignore environment-owned caches outside tracked repository scope rather than deleting user artifacts.

- [ ] **Step 3: Add the exact Methods paragraph without numerical claims**

Replace the comparator paragraph in `paper/manuscript.tex` with:

```tex
The prespecified same-input denominator schedules observed counts, a capacity-matched
autoencoder, ALRA, MAGIC, DCA, scVI, SAVER, scZiva, afMF, BiAEImpute, scCR,
and scSDAE as a fixed twelve-method denominator. Before final evaluation, the
development plan will evaluate thirty-four complete comparator configurations
across all sixteen development datasets and three model seeds. Each tunable method varies one
prespecified, scientifically interpretable axis around its upstream default;
ALRA rank and SAVER empirical-Bayes parameters remain automatic. The selector
will average model seeds within technical view and pair views within biological draw,
then will select one global configuration per method using six reconstruction
metrics, all-eligible Pareto filtering, and a deterministic Pareto-only
quarter-rank tuple. Comparator selection cannot access MaskImpute performance,
downstream endpoints, or final data by design. Selected payloads and method
identities will be frozen unchanged for candidate assessment, final evaluation,
trajectory evaluation, and scaling. Methods with no eligible development
configuration will remain in the scheduled denominator with their complete
reason-coded failure counts and will be excluded only from numerical estimands
for which no selected configuration exists.
```

Retain existing citations and attach them to the appropriate method names. Do not add a result, performance adjective, superiority statement, or Genome Biology fitness claim before sealed evidence exists.

Add verified BibTeX entries to `paper/references.bib` and cite them at the first
method mention using these registry-bound DOIs: scZiva
`10.1186/s12859-026-06422-2`, afMF `10.1002/ctm2.70283`, BiAEImpute
`10.1186/s12864-025-11988-x`, scCR `10.52202/079017-0598`, and scSDAE
`10.3390/genes11050532`. Transcribe title/authors/year/journal from the DOI
metadata; require the DOI in each entry and do not infer missing fields. Extend
the documentation test to assert each DOI and citation key occurs, and fail if
the LaTeX log contains `undefined citations`, `Citation ... undefined`, or
`There were undefined references`.

- [ ] **Step 4: Update workflow and checklists**

Document the exact command order without executing it:

```text
1. implement/review the separate calibration amendment;
2. python scripts/run_comparator_tuning_smoke.py;
3. python scripts/run_development_competition.py;
4. python scripts/select_comparator_configurations.py;
5. python scripts/build_development_selection_input.py;
6. python scripts/promote_development_selection_input.py;
7. python scripts/select_development_candidate.py;
8. execute the fixed revision command for v28/v29 only when its activation receipt triggers;
9. python scripts/freeze_publication_round.py prepare;
10. commit and review study/frozen_method.json plus its bound authorities;
11. python scripts/freeze_publication_round.py freeze "$ROUND_DIR";
12. python scripts/run_frozen_final.py "$ROUND_DIR" \
      --simulator-assets-root "$SIMULATOR_ASSETS_ROOT" \
      --simulator-r-environment "$SIMULATOR_R_ENVIRONMENT".
```

The three uppercase paths are not scientific overrides: the reviewed release
operator sets them to the newly opened round and the separately pinned external
simulator assets/environment, then records their receipts. Document the exact
revision CLI after Task 13 from `revision_commands.py --help`; do not replace it
with a free-form rerun.

Add checklist items for exact 2,896 rows, all 34 smoke completions, readiness (controls + five established + at least three modern), BiAEImpute status, selected payload hashes, 1,760 final rows, 44 trajectory rows, complete execution-status table, and no claim against unavailable methods.

- [ ] **Step 5: Close static hygiene rules**

Make hygiene tests inspect tracked/unignored repository paths using `git ls-files --cached --others --exclude-standard`; reject `__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.tmp`, partial transaction files, comparator smoke/selection outputs, checkpoints, matrices, and publication assets unless they are the explicitly expected committed authority/spec/manuscript files. Do not delete or mutate artifacts in tests.

Delete and stage the eight already tracked
`DenseLayerPack/__pycache__/*.cpython-310.pyc` files enumerated by
`git ls-files | rg '(__pycache__|\\.pyc$)'`. For ignored scientific outputs,
the test performs fixed `os.path.lexists` checks at comparator smoke/selection,
competition checkpoint, current final-round, `paper/generated`, and
`paper/figures` roots; the tracked/unignored scan alone cannot see ignored
files. Allow committed `historical/` and `feedback/` evidence and do not apply a
broad PDF/figure ban.

After staging the tracked `.pyc` deletions, remove only generated ignored cache
content from the isolated worktree with these bounded commands:

```bash
find maskimpute maskimpute_benchmark scripts tests DenseLayerPack \
  -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find maskimpute maskimpute_benchmark scripts tests DenseLayerPack \
  -depth -type d -name '__pycache__' -empty -delete
for cache in .pytest_cache .ruff_cache; do
  if test -d "$cache" && test ! -L "$cache"; then
    find "$cache" -depth -delete
  fi
done
```

The paths are fixed generated-cache roots inside this worktree; do not broaden
the command to `artifacts/`, `historical/`, `feedback/`, or user directories.

- [ ] **Step 6: Run docs, hygiene, and LaTeX checks**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_publication_repository_hygiene.py tests/test_comparator_tuning.py \
  -q -W error -p no:cacheprovider
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/manuscript.tex
latexmk -cd -C paper/manuscript.tex
```

Expected: tests PASS, `paper/manuscript.pdf` builds without undefined references,
and the explicit `latexmk -cd -C` command removes only that build's ignored
products. The repository has no cleanup target; do not claim or invoke one.

- [ ] **Step 7: Commit Genome Biology documentation and hygiene**

```bash
git add docs/development-selection-workflow.md paper/manuscript.tex paper/references.bib \
  paper/submission_checklist.md docs/genome-biology-submission-checklist.md \
  tests/test_publication_repository_hygiene.py tests/test_comparator_tuning.py
git add -u DenseLayerPack/__pycache__
git commit -m "docs: describe fair comparator selection"
```

---

### Task 18: Run cross-scope verification and independent review

**Files:**

- Modify only if a verification failure identifies an in-scope defect; use the owning task's exact source/test file and a focused fix commit.
- Review: all files listed in Tasks 1-17.

**Interfaces:**

- Consumes: the complete branch at the Task 17 commit.
- Produces: fresh test/lint/compile/LaTeX evidence, exact denominator audit output, a clean diff, and an independent review decision. It does not create scientific evidence.

- [ ] **Step 1: Prove exact static authority and denominator values without execution**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py::test_tracked_authority_has_exact_grid_and_operational_contract \
  tests/test_benchmark_runner.py::test_tracked_plan_has_exact_2896_rows_and_complete_comparator_blocks \
  tests/test_benchmark_runner.py::test_revision_plan_contains_exactly_one_48_row_maskimpute_candidate \
  tests/test_final_runner.py::test_final_plan_is_1760_and_all_selectable_split_is_1480_280 \
  tests/test_final_runner.py::test_unavailable_stochastic_comparator_reclassifies_120_seeded_rows \
  tests/test_final_runner.py::test_trajectory_always_has_44_rows_and_preserves_three_unavailable_seeds \
  -q -W error -p no:cacheprovider
```

Expected: six tests PASS using only tracked authority and synthetic in-memory
bindings; no dataset H5AD, checkpoint, smoke receipt, or scientific result is
loaded or created.

- [ ] **Step 2: Run focused suites in dependency order**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py tests/test_selection_authority.py \
  -q -W error -p no:cacheprovider
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_benchmark_runner.py tests/test_prezero_evidence.py \
  tests/test_runtime_environments.py -q -W error -p no:cacheprovider
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_development_evaluation.py tests/test_candidate_selection.py \
  tests/test_selection_promotion.py tests/test_revision_authority.py \
  tests/test_revision_evaluation.py -q -W error -p no:cacheprovider
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py tests/test_final_runner.py \
  tests/test_scaling_panel.py tests/test_downstream_evidence.py \
  tests/test_final_analysis.py tests/test_publication_synthesis.py \
  tests/test_publication_repository_hygiene.py \
  -q -W error -p no:cacheprovider
```

Expected: every command exits 0; only pre-existing environment-availability skips are allowed.

- [ ] **Step 3: Run the full hermetic suite**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  -q -W error -p no:cacheprovider
```

Expected: exit 0 with no new skip, xfail, warning, or collection failure caused by this work.

- [ ] **Step 4: Run static, compile, diff, and manuscript gates**

```bash
/tmp/maskimpute-supported/bin/ruff check --no-cache maskimpute_benchmark scripts tests
/tmp/maskimpute-supported/bin/ruff format --check maskimpute_benchmark scripts tests
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/maskimpute-comparator-plan-pyc \
  /tmp/maskimpute-supported/bin/python -m compileall -q \
  maskimpute_benchmark scripts tests
git diff --check cd99104..HEAD
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/manuscript.tex
latexmk -cd -C paper/manuscript.tex
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_publication_repository_hygiene.py -q -W error -p no:cacheprovider
git status --short
if git status --short --ignored | \
  rg '(__pycache__|\.pytest_cache|\.ruff_cache|comparator_(smoke|selection)\.json|manuscript\.(aux|bbl|blg|fdb_latexmk|fls|log|out|pdf))'; then
  exit 1
fi
```

Expected: lint/format/compile/diff/LaTeX/hygiene exit 0; cleanup precedes the
final status checks; `git status --short` is empty in this isolated worktree;
and the ignored-path filter finds no cache, comparator evidence, or manuscript
build product.

- [ ] **Step 5: Perform an independent two-pass review**

Invoke `superpowers:requesting-code-review`. First pass checks exact spec coverage, scientific independence, cardinalities, status policy, hash cycles, identity domains, and no-default propagation. Second pass checks implementation quality, unsafe I/O, race/idempotence, checkpoint replay, private-path leakage, tests, and documentation. Resolve every blocking finding with a failing regression and a focused fix commit, then rerun Steps 2-4.

- [ ] **Step 6: Confirm the completion boundary**

Verify all of the following are true before saying implementation is complete:

```text
- authority, typed dispatch, budget, storage, smoke gate, selector, receipt,
  readiness, candidate projection, revisions, freeze, final, trajectory,
  scaling, downstream analysis, documentation, and hygiene are integrated;
- no development/final scientific run or generated comparator receipt was made;
- calibration repair is still the next blocked scientific subproject;
- publication results, release/license, and asset export remain pending;
- the branch is clean and independently approved.
```

Do not create a final empty commit. If review fixes were required, their focused commits are the Task 18 deliverable; otherwise the fresh verification and clean review decision are the deliverable.

---

## Plan Self-Review Checklist

- [ ] **Spec coverage:** Map tracked authority/grid to Tasks 1-3; identities/plan/dispatch/budget/storage/smoke to Tasks 4-9; collapse/selection/receipt to Tasks 10-11; candidate projection/readiness to Task 12; revision reuse to Task 13; freeze to Task 14; final/trajectory to Task 15; scaling/downstream/claims to Task 16; Genome Biology documentation/hygiene to Task 17; full verification/review to Task 18.
- [ ] **Hash direction:** Confirm the acyclic chain is `methods.json -> comparator_tuning.json -> selection_contract.json -> development_search.json -> runner plan/checkpoint -> comparator selection -> candidate selection -> revision -> freeze -> final/trajectory/scaling -> analysis/assets`.
- [ ] **Identity types:** Confirm `configuration_payload_sha256` is the adapter-payload hash, `configuration_method_identity_sha256` is the preselection stable identity, `selected_method_identity_sha256` repeats it exactly, and `nonexecution_identity_sha256` is used only when no configuration was selected.
- [ ] **Cardinalities:** Confirm tests assert 34 configurations, 2,896 development rows, 1,632 comparator rows, 48 rows per comparator config, 48 rows per revision, 1,760 final rows, 1,480/280 all-selectable split, and 44 trajectory rows.
- [ ] **No placeholders:** Run the scan below and rewrite any matching prose as an exact action/code block:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /tmp/maskimpute-supported/bin/python - <<'PY'
from pathlib import Path
text = Path("docs/superpowers/plans/2026-07-18-fair-comparator-tuning.md").read_text().lower()
terms = ["tb" + "d", "to" + "do", "implement " + "later", "fill " + "in details", "appropriate " + "error handling", "similar " + "to task"]
found = [term for term in terms if term in text]
assert not found, found
PY
```

Expected: exit 0 and no output.

- [ ] **Type consistency:** Verify every later consumer uses the exact names declared here: `ComparatorTuningAuthority`, `BoundComparatorConfiguration`, `ComparatorSelectionProjection`, `SelectedComparatorConfiguration`, `configuration_payload_sha256`, `configuration_method_identity_sha256`, `selected_method_identity_sha256`, and `nonexecution_identity_sha256`.
- [ ] **Plan structure and paths:** Run this static audit and fix every failure:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /tmp/maskimpute-supported/bin/python - <<'PY'
from pathlib import Path
import re

path = Path("docs/superpowers/plans/2026-07-18-fair-comparator-tuning.md")
text = path.read_text()
tasks = [int(value) for value in re.findall(r"^### Task (\d+):", text, re.M)]
assert tasks == list(range(1, 19)), tasks
allowed_new = {"tests/test_comparator_tuning.py"}
referenced = set(re.findall(r"tests/[A-Za-z0-9_./-]+\.py", text))
missing = sorted(name for name in referenced if not Path(name).exists() and name not in allowed_new)
assert not missing, missing
assert "test_" + "evaluation_manifest.py" not in text
assert "Obs" + "Covariate" not in text
assert "v28-c01-nb-" + "calibrated-r1-g1" not in text
PY
```

- [ ] **Fixture/API consistency:** For every named test fixture/helper in Tasks
  6--17, confirm the same step either defines it or names the existing
  file-local factory it extends. Confirm public function names with `rg '^def '
  maskimpute_benchmark` and dry-run every listed pytest node with `--collect-only`
  before implementation review.
- [ ] **Asset boundary:** Confirm stale descendant/asset bindings are rejected
  in Task 16, while actual figure/table export and asset-tamper publication tests
  remain explicitly assigned to the separate asset subproject rather than
  falsely marked complete here.
- [ ] **Completion boundary:** Confirm the plan implements infrastructure only and cannot run the scientific competition until a separate calibration amendment and a real comparator smoke receipt are reviewed.
