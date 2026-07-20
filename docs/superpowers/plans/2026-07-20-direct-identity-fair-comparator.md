# Direct-Identity Fair-Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace content-digest identity in the fair-comparator segment with explicit typed snapshots and direct equality while preserving the approved 34-configuration grid, 2,896-row scientific denominator, exact budget replay, and all unrelated legacy behavior.

**Architecture:** Keep legacy publication provenance unchanged and isolate the new direct-identity path in three focused modules: plan construction, execution records, and checkpoint replay. `comparator_tuning.py` remains the strict authority/codec boundary. Shared development entry points select the direct path only for the fair-comparator base plan and hand a plain selected-comparator map back to legacy publication consumers.

**Tech Stack:** Python 3.11, frozen dataclasses, closed canonical JSON, NumPy, pytest, Ruff, Git.

## Global Constraints

- The governing design is `docs/superpowers/specs/2026-07-20-direct-identity-fair-comparator-design.md` at commit `958790e`.
- Fair-comparator code and schemas must not compute, emit, require, validate, rename, or disguise content digests.
- The tokens `hash`, `digest`, `checksum`, `fingerprint`, `sha`, and algorithm variants are forbidden in fair-comparator field names and generated artifacts.
- Direct equality compares actual typed values or complete canonical encoded bytes.
- The exact configuration grid remains 34 rows: eight four-point grids plus the ALRA and SAVER defaults.
- The base scientific denominator remains 2,896 rows: 16 observed, 48 capacity-matched autoencoder, 1,200 MaskImpute, and 1,632 comparator rows.
- Comparator configurations share one method-level budget; MaskImpute keeps separate method-and-kind scopes.
- Intrinsic comparator terminal statuses are exactly `failed`, `timeout`, `resource_exceeded`, and `unavailable`.
- Blocking statuses are exactly `budget_exhausted`, `blocked_authority`, and `infrastructure_error`.
- Durable records are not retried; only unfinished transaction intent may recover.
- Old and mixed fair-comparator schemas fail closed and are never upgraded in place.
- Existing legacy dataset, calibration, revision, final, scaling, and archive provenance remains unchanged outside the direct segment.
- No real comparator, smoke, tuning, evaluator, competition, final, or other scientific workload runs during implementation.
- Use `env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python` for focused tests; do not add interpreter-environment flags rejected by spawn tests.
- Every production change starts with a failing regression and ends with focused plus adjacent warning-strict tests.

## Relationship to the 2026-07-18 plan

Execute this plan immediately after fair-comparator Task 7 commit `869938f` and
before Task 8 of `2026-07-18-fair-comparator-tuning.md`. When this migration is
accepted, re-review Task 7 over `22b02d7..HEAD`, then resume Tasks 8–18.

The scientific requirements in Tasks 8–18 remain binding. Their identity
mechanisms are superseded as follows:

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

Future task implementers must remove rather than reproduce any superseded field
shown in old snippets. Task reviews apply this table before the older task text.

## File Structure and Responsibility Map

- Modify `maskimpute_benchmark/comparator_tuning.py`: direct authority parsing, typed payload codec, method binding, selected-map types.
- Create `maskimpute_benchmark/fair_comparator_plan.py`: explicit input descriptors, complete run identities, 2,896-row plan construction.
- Create `maskimpute_benchmark/fair_comparator_execution.py`: direct requests, direct records, comparator dispatch, bounded log receipts.
- Create `maskimpute_benchmark/fair_comparator_checkpoint.py`: transaction intent, exact prefix validation, central budget replay, completeness derivation.
- Modify `maskimpute_benchmark/selection.py`: path/schema/revision linkage and direct selected-map projection only.
- Modify `maskimpute_benchmark/revisions.py`: carry the direct comparator reference unchanged through candidate-only revisions.
- Modify `maskimpute_benchmark/runner.py`: route only the fair-comparator base plan through the three new modules; preserve legacy modes.
- Modify `maskimpute_benchmark/development_evaluation.py` and `maskimpute_benchmark/downstream_evidence.py`: accept the direct checkpoint schema and selected map.
- Modify `study/comparator_tuning.json`, `study/selection_contract.json`, and `study/development_search.json`: direct comparator authority references.
- Create `tests/test_fair_comparator_plan.py`, `tests/test_fair_comparator_execution.py`, and `tests/test_fair_comparator_checkpoint.py`: isolated direct-path tests.
- Modify existing comparator, runner, selection, revision, development-evaluation, pre-zero, and downstream tests for schema closure and legacy non-regression.

---

### Task 1: Convert the comparator authority and codec to direct identity

**Files:**

- Modify: `study/comparator_tuning.json`
- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Modify: `tests/test_comparator_tuning.py`

**Interfaces:**

- Produces: `AUTHORITY_REVISION`, `ComparatorConfiguration`, `ComparatorTuningAuthority.authority_revision`, `decode_comparator_configuration(method_id, payload)`, `parse_comparator_tuning_authority(payload, *, registry)`, and `load_comparator_tuning_authority(repository, *, registry, require_clean=True)`.
- Preserves: the exact expected configuration table, policy constants, strict canonical JSON parsing, and the 34 typed adapter dataclasses.

- [ ] **Step 1: Write the failing tracked-schema regression**

Add recursive key inspection and exact signature checks:

```python
import inspect


FORBIDDEN_IDENTITY_TOKENS = (
    "hash", "digest", "checksum", "fingerprint", "sha",
)


def _all_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key
            for child in value.values()
            for key in _all_keys(child)
        )
    if isinstance(value, list):
        return tuple(key for child in value for key in _all_keys(child))
    return ()


def test_tracked_comparator_authority_uses_only_direct_identity() -> None:
    payload = json.loads((ROOT / "study/comparator_tuning.json").read_text())
    assert payload["authority_revision"] == "fair-comparator-direct-v1"
    assert not any(
        token in key.lower()
        for key in _all_keys(payload)
        for token in FORBIDDEN_IDENTITY_TOKENS
    )
    parameters = inspect.signature(decode_comparator_configuration).parameters
    assert tuple(parameters) == ("method_id", "payload")
```

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py::test_tracked_comparator_authority_uses_only_direct_identity \
  -q -W error -p no:cacheprovider
```

Expected: fail because `authority_revision` is absent and superseded fields are present.

- [ ] **Step 3: Replace the public authority types and decoder signature**

Use these exact public shapes:

```python
AUTHORITY_REVISION = "fair-comparator-direct-v1"


@dataclass(frozen=True, slots=True)
class ComparatorConfiguration:
    method_id: str
    configuration_id: str
    payload_json: str
    is_upstream_default: bool

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.payload_json))

    def decode(self) -> ComparatorAdapterConfig:
        return decode_comparator_configuration(self.method_id, self.payload)


@dataclass(frozen=True, slots=True)
class ComparatorTuningAuthority:
    schema_version: int
    contract_id: str
    authority_revision: str
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
```

Change the decoder to:

```python
def decode_comparator_configuration(
    method_id: str,
    payload: Mapping[str, object],
) -> ComparatorAdapterConfig:
    config_type = _CONFIG_TYPES.get(method_id)
    if config_type is None or type(payload) not in {dict, MappingProxyType}:
        raise ComparatorTuningError("comparator method or payload is invalid")
    defaults = _json_payload(config_type())
    observed = dict(payload)
    if set(observed) != set(defaults):
        raise ComparatorTuningError(
            "comparator payload differs from its complete field set"
        )
    constructor = _decode_closed_primitive_fields(method_id, observed, defaults)
    decoded = config_type(**constructor)
    if encode_comparator_configuration(decoded) != observed:
        raise ComparatorTuningError(
            "comparator payload changed during typed normalization"
        )
    return decoded
```

Use this complete primitive decoder before constructing a dataclass:

```python
def _decode_closed_primitive_fields(
    method_id: str,
    observed: Mapping[str, object],
    defaults: Mapping[str, object],
) -> dict[str, object]:
    constructor: dict[str, object] = {}
    for name, default in defaults.items():
        value = observed[name]
        if method_id == "dca" and name == "hidden_size":
            if (
                type(value) is not list
                or not value
                or any(type(item) is not int or item <= 0 for item in value)
            ):
                raise ComparatorTuningError(
                    "DCA hidden_size must be a positive-integer JSON array"
                )
            constructor[name] = tuple(value)
            continue
        if type(value) is float and (
            not math.isfinite(value)
            or (value == 0.0 and math.copysign(1.0, value) < 0.0)
        ):
            raise ComparatorTuningError(
                f"comparator field {name} has an invalid float value"
            )
        if not _primitive_type_matches(value, default):
            raise ComparatorTuningError(
                f"comparator field {name} has the wrong primitive type"
            )
        constructor[name] = value
    return constructor
```

- [ ] **Step 4: Convert the tracked JSON and strict parser**

Set `schema_version` to `2`, add:

```json
"authority_revision": "fair-comparator-direct-v1"
```

Remove all configuration-level and document-level superseded fields. Change
the parser signature to:

```python
def parse_comparator_tuning_authority(
    payload: object,
    *,
    registry: MethodRegistry,
) -> ComparatorTuningAuthority:
```

Require the exact schema-2 key set and compare every row's canonical bytes with
`_EXPECTED_CONFIGURATION_PAYLOADS`. Detect duplicate payloads within a method
using `payload_json` strings. Change the clean loader to parse the owned regular
file and call the new parser without calculating file content summaries.

- [ ] **Step 5: Add direct mutation and old-schema rejection tests**

Cover signed negative zero, Unicode representation drift, payload mutation,
row reorder, a duplicate payload under another ID, old top-level fields, and a
mixed schema. Each must raise `ComparatorTuningError`; all 34 normative rows
must decode and re-encode exactly.

- [ ] **Step 6: Run focused and adjacent tests**

Run:

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py tests/test_core_method_adapters.py \
  tests/test_priority_method_adapters.py tests/test_required_legacy_method_adapters.py \
  -q -W error -p no:cacheprovider
```

Expected: all pass, with only pre-existing availability skips.

- [ ] **Step 7: Commit**

```bash
git add study/comparator_tuning.json maskimpute_benchmark/comparator_tuning.py tests/test_comparator_tuning.py
git commit -m "refactor: use direct comparator authority identity"
```

### Task 2: Bind tracked references and methods by explicit fields

**Files:**

- Modify: `study/selection_contract.json`
- Modify: `study/development_search.json`
- Modify: `maskimpute_benchmark/comparator_tuning.py`
- Modify: `maskimpute_benchmark/selection.py`
- Modify: `maskimpute_benchmark/revisions.py`
- Modify: `tests/test_comparator_tuning.py`
- Modify: `tests/test_selection_authority.py`
- Modify: `tests/test_revision_authority.py`
- Modify: `tests/test_method_registry.py`

**Interfaces:**

- Produces: `ComparatorAuthorityReference`, `ComparatorMethodBinding`, `BoundComparatorConfiguration`, `comparator_method_binding(method_spec)`, and `bind_comparator_configuration_identity(configuration, method_spec, authority)`.
- Changes `SelectionAuthority` and revision projection to carry a direct comparator reference rather than comparator file/payload summaries.

- [ ] **Step 1: Write the failing direct-linkage tests**

```python
def test_selection_contract_binds_comparator_authority_by_direct_reference() -> None:
    contract = json.loads((ROOT / "study/selection_contract.json").read_text())
    reference = contract["comparator_tuning"]
    assert reference == {
        "path": "study/comparator_tuning.json",
        "schema_version": 2,
        "authority_revision": "fair-comparator-direct-v1",
    }


def test_bound_comparator_contains_full_method_projection() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )
    row = authority.configurations_for("magic")[0]
    bound = bind_comparator_configuration_identity(
        row, registry.by_id("magic"), authority
    )
    assert bound.configuration == row
    assert bound.authority_reference.authority_revision == authority.authority_revision
    assert bound.method == comparator_method_binding(registry.by_id("magic"))
```

- [ ] **Step 2: Run the tests and verify RED**

Run the two named tests with warning-strict pytest. Expected: fail because the
tracked reference and direct binding types do not exist.

- [ ] **Step 3: Add exact direct binding types**

```python
@dataclass(frozen=True, slots=True)
class ComparatorAuthorityReference:
    path: str
    schema_version: int
    authority_revision: str


@dataclass(frozen=True, slots=True)
class ComparatorMethodBinding:
    method_id: str
    execution_scope: str
    integration_status: str
    adapter_key: str
    environment_id: str
    environment_status: str
    source_kind: str
    source_url: str | None
    source_revision: str | None
    source_tree: str | None
    source_freeze_binding: str | None
    gpu_mode: str
    timeout_seconds: int
    max_rss_gib: int | float
    max_gpu_gib: int | float


@dataclass(frozen=True, slots=True)
class BoundComparatorConfiguration:
    configuration: ComparatorConfiguration
    authority_reference: ComparatorAuthorityReference
    method: ComparatorMethodBinding
```

`comparator_method_binding` copies the named fields from `MethodSpec` and uses
`method_spec.id` as `adapter_key`. `bind_comparator_configuration_identity`
requires one equal authority row, a matching method ID, canonical payload
bytes, and successful typed decode, then returns the three actual values.

- [ ] **Step 4: Convert both tracked references and their parsers**

Replace the two comparator-specific fields in each tracked document with:

```json
"comparator_tuning": {
  "path": "study/comparator_tuning.json",
  "schema_version": 2,
  "authority_revision": "fair-comparator-direct-v1"
}
```

Remove `study/comparator_tuning.json` from generic file-summary maps used only
to bind legacy authorities. `load_selection_authority` must validate the exact
reference, load the authority, and compare all three reference fields plus the
fully parsed authority. `revisions.py` copies the reference object unchanged.

- [ ] **Step 5: Add recomputed-summary and projection-drift regressions**

Mutate a method projection field, authority revision, authority path, schema
version, full payload, and row order. Re-encode each whole fixture coherently.
Every case must still be rejected through direct validation. Prove unrelated
legacy authority fields remain byte-for-byte unchanged.

- [ ] **Step 6: Run focused and adjacent tests**

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py tests/test_selection_authority.py \
  tests/test_revision_authority.py tests/test_method_registry.py \
  -q -W error -p no:cacheprovider
```

- [ ] **Step 7: Commit**

```bash
git add study/selection_contract.json study/development_search.json \
  maskimpute_benchmark/comparator_tuning.py maskimpute_benchmark/selection.py \
  maskimpute_benchmark/revisions.py tests/test_comparator_tuning.py \
  tests/test_selection_authority.py tests/test_revision_authority.py \
  tests/test_method_registry.py
git commit -m "refactor: bind comparator methods by explicit fields"
```

### Task 3: Build the full development plan from complete run identities

**Files:**

- Create: `maskimpute_benchmark/fair_comparator_plan.py`
- Modify: `maskimpute_benchmark/runner.py`
- Create: `tests/test_fair_comparator_plan.py`
- Modify: `tests/test_benchmark_runner.py`

**Interfaces:**

- Produces: `PreparedInputDescriptor`, `DirectAuthorizedConfiguration`, `ComparatorRunIdentity`, `DirectPlanEntry`, `DirectCompetitionPlan`, `describe_prepared_input`, and `build_direct_competition_plan`.
- Consumes the same registry, 16 dataset bindings, runner authority, and 61 authorized configurations used by the accepted 2,896-row plan.

- [ ] **Step 1: Write the failing plan-shape and schema tests**

```python
def test_direct_plan_has_exact_denominator_and_no_summary_fields() -> None:
    plan = _build_direct_plan_fixture()
    assert plan.identity_mode == "direct-v1"
    assert len(plan.entries) == 2_896
    assert len(plan.configurations) == 61
    assert {
        "observed": sum(row.identity.method_id == "observed" for row in plan.entries),
        "capacity": sum(
            row.identity.method_id == "capacity-matched-ae" for row in plan.entries
        ),
        "maskimpute": sum(
            row.identity.method_id == "maskimpute" for row in plan.entries
        ),
        "comparators": sum(
            row.identity.configuration_kind == "comparator_tuning"
            for row in plan.entries
        ),
    } == {"observed": 16, "capacity": 48, "maskimpute": 1_200, "comparators": 1_632}
    encoded = plan.to_dict()
    assert not _contains_forbidden_identity_key(encoded)
```

- [ ] **Step 2: Run the test and verify RED**

Expected: import failure for `maskimpute_benchmark.fair_comparator_plan`.

- [ ] **Step 3: Add the exact direct plan dataclasses**

```python
@dataclass(frozen=True, slots=True)
class PreparedInputDescriptor:
    dataset_id: str
    source_reference: str
    preprocessing_revision: str
    shape: tuple[int, int]
    dtype: str
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    batch_labels: tuple[str, ...]
    total_count: float
    nonzero_count: int
    minimum: float
    maximum: float
    mechanism: str
    mask_seed: int
    technical_view: str


@dataclass(frozen=True, slots=True)
class DirectAuthorizedConfiguration:
    method: ComparatorMethodBinding
    configuration_id: str
    configuration_kind: str
    payload: tuple[tuple[str, object], ...]
    requires_count_score: bool
    requires_calibration: bool


@dataclass(frozen=True, slots=True)
class ComparatorRunIdentity:
    workflow_schema: str
    authority_revision: str
    ordinal: int
    method: ComparatorMethodBinding
    configuration_id: str
    configuration_kind: str
    configuration_payload: tuple[tuple[str, object], ...]
    dataset_id: str
    mechanism: str
    biological_id: str
    technical_view: str
    mask_seed: int
    model_seed: int | None
    draw_index: int


@dataclass(frozen=True, slots=True)
class DirectPlanEntry:
    run_id: str
    identity: ComparatorRunIdentity
    preflight_status: Literal["planned", "blocked_authority"]
    preflight_reason: str | None
    requires_count_score: bool
    requires_calibration: bool


@dataclass(frozen=True, slots=True)
class DirectCompetitionPlan:
    schema_version: int
    identity_mode: Literal["direct-v1"]
    authority_revision: str
    inputs: tuple[PreparedInputDescriptor, ...]
    entries: tuple[DirectPlanEntry, ...]
    configurations: tuple[DirectAuthorizedConfiguration, ...]

    def to_dict(self) -> dict[str, object]:
        return _direct_plan_to_json(self)
```

Use canonical tuples for payloads but recursively restore JSON lists and
objects in `to_dict`. `describe_prepared_input` copies all ordered IDs and
labels and computes only the numeric invariants named in the dataclass.

- [ ] **Step 4: Port the accepted ordering and denominator logic**

Build 61 configurations and 2,896 entries in the already accepted order. Use
configuration IDs, not content summaries, as the budget configuration key. A
run ID is:

```python
def direct_run_id(identity: ComparatorRunIdentity) -> str:
    seed = "deterministic" if identity.model_seed is None else f"seed-{identity.model_seed}"
    return (
        f"run-{identity.ordinal:04d}-{identity.method.method_id}-"
        f"{identity.dataset_id.removeprefix('dataset-')}-{seed}-"
        f"{identity.configuration_id}"
    )
```

Validate uniqueness, contiguous ordinals, 34 ordered comparator blocks of 48,
and the exact component counts before returning. Revision v28/v29 construction
continues to produce 48 candidate-only entries using the same direct types.

- [ ] **Step 5: Route the fair-comparator base plan without changing legacy plans**

Add a narrow `build_fair_comparator_plan(...) -> DirectCompetitionPlan`
entry point in `runner.py` that delegates to the new module. Do not alter
legacy `CompetitionPlan` serialization or its existing callers.

- [ ] **Step 6: Run focused and adjacent plan tests**

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_fair_comparator_plan.py tests/test_benchmark_runner.py \
  -k 'direct or plan or 2896 or comparator' \
  -q -W error -p no:cacheprovider
```

- [ ] **Step 7: Commit**

```bash
git add maskimpute_benchmark/fair_comparator_plan.py maskimpute_benchmark/runner.py \
  tests/test_fair_comparator_plan.py tests/test_benchmark_runner.py
git commit -m "feat: build direct fair-comparator plans"
```

### Task 4: Dispatch direct requests and retain digest-free run records

**Files:**

- Create: `maskimpute_benchmark/fair_comparator_execution.py`
- Modify: `maskimpute_benchmark/runner.py`
- Create: `tests/test_fair_comparator_execution.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_priority_method_adapters.py`

**Interfaces:**

- Produces: `DirectExecutionRequest`, `DirectLogReceipt`, `DirectRunResult`, `DirectMetricRow`, `DirectPreZeroEvidence`, `DirectEvaluatedAttempt`, `create_direct_request`, and `execute_direct_request`.
- Consumes: `DirectPlanEntry`, one `PreparedDataset`, the exact authority row, and existing adapter callables.

- [ ] **Step 1: Write the failing all-adapter direct-dispatch test**

Use spies for all ten comparator adapters. For every tracked row, create a
direct request, dispatch it, and assert the spy receives `row.decode()`. Assert
the serialized request and result contain no forbidden identity key and the log
receipt equals:

```python
{
    "stream": "stdout",
    "original_byte_count": 3,
    "capture_policy": "discard_content",
    "terminal_reason": None,
}
```

- [ ] **Step 2: Run the test and verify RED**

Expected: import failure for `maskimpute_benchmark.fair_comparator_execution`.

- [ ] **Step 3: Add exact direct request and record types**

```python
@dataclass(frozen=True, slots=True)
class DirectExecutionRequest:
    identity: ComparatorRunIdentity
    method_spec: MethodSpec
    method_input: MethodInput
    timeout_seconds: float
    max_rss_bytes: int
    max_gpu_bytes: int


@dataclass(frozen=True, slots=True)
class DirectLogReceipt:
    stream: Literal["stdout", "stderr"]
    original_byte_count: int
    capture_policy: Literal["discard_content"]
    terminal_reason: str | None


@dataclass(frozen=True, slots=True)
class DirectRunResult:
    run_id: str
    identity: ComparatorRunIdentity
    status: str
    reason: str | None
    runtime_seconds: int | float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    rss_measurement: str
    gpu_measurement: str
    excluded_cell_count: int
    excluded_cell_ids: tuple[str, ...]
    retained_cell_count: int
    retained_cell_ids: tuple[str, ...]
    retained_gene_count: int
    observed_zero_count: int
    stdout: DirectLogReceipt
    stderr: DirectLogReceipt


@dataclass(frozen=True, slots=True)
class DirectMetricRow:
    identity: ComparatorRunIdentity
    metric: str
    value: float | None
    n: int
    status: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class DirectPreZeroEvidence:
    applicable: bool
    status: str
    reason: str | None
    shape: tuple[int, int] | None
    dtype: Literal["<f8"] | None
    encoding: Literal["zlib"] | None
    path: str | None
    compressed_byte_count: int


@dataclass(frozen=True, slots=True)
class DirectEvaluatedAttempt:
    run: DirectRunResult
    metrics: tuple[DirectMetricRow, ...]
    native_output: np.ndarray | None
    native_output_scale: str | None
    evaluator_output: np.ndarray | None
    p_pre_zero_evidence: DirectPreZeroEvidence
```

The checkpoint serializer retains `run`, `metrics`, and the existing scientific
pre-zero shape/storage values needed by the evaluator, but omits in-memory
matrices and every legacy output-identity field. Reopening compressed pre-zero
data validates the owned regular path, exact byte count, zlib decoding, dtype,
and shape before use.

- [ ] **Step 4: Implement exact resolution and post-attempt revalidation**

`create_direct_request` requires complete equality between the plan identity,
authority row, method projection, prepared input descriptor, and method spec.
`execute_direct_request` resolves exactly one row, decodes it, invokes the
matching adapter with its typed config, and re-encodes the effective config for
complete equality. Preserve the accepted status mapping and resource limits.
Do not call `ExecutionRequest.create` or serialize `RawRunResult` on this path.

- [ ] **Step 5: Add closure and failure-path tests**

Cover all ten adapters, wrong configuration labels, payload drift before and
after dispatch, method projection drift, unavailable adapters, timeouts,
resource excess, infrastructure failure, duplicate authority matches, unknown
fields, and registry-default rejection. Test only spies/fake outcomes.

- [ ] **Step 6: Run focused and adjacent tests**

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_fair_comparator_execution.py tests/test_benchmark_runner.py \
  tests/test_priority_method_adapters.py \
  -k 'direct or comparator or typed or dispatch' \
  -q -W error -p no:cacheprovider
```

- [ ] **Step 7: Commit**

```bash
git add maskimpute_benchmark/fair_comparator_execution.py maskimpute_benchmark/runner.py \
  tests/test_fair_comparator_execution.py tests/test_benchmark_runner.py \
  tests/test_priority_method_adapters.py
git commit -m "feat: dispatch direct comparator requests"
```

### Task 5: Replay direct checkpoints and budgets exactly

**Files:**

- Create: `maskimpute_benchmark/fair_comparator_checkpoint.py`
- Modify: `maskimpute_benchmark/runner.py`
- Create: `tests/test_fair_comparator_checkpoint.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_prezero_evidence.py`
- Modify: `tests/test_development_evaluation.py`

**Interfaces:**

- Produces: `DirectDevelopmentBudget`, `DirectCheckpointReport`, `DirectCheckpointStore`, `replay_direct_development_budget`, and `direct_comparator_selection_status`.
- Preserves Task 7's exact configuration-limit policy, method budget scopes, terminal/blocking partition, and transaction recovery rule.

- [ ] **Step 1: Write failing direct checkpoint regressions**

```python
def test_direct_checkpoint_replays_exact_prefix_and_budget(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    report = store.write(
        plan,
        _terminal_prefix(plan, 3),
        registry=registry,
        prepared_datasets=prepared,
    )
    assert report.plan_snapshot == plan.to_dict()
    assert report.budget == replay_direct_development_budget(
        registry, plan.entries, report.records
    ).to_dict()
    assert not _contains_forbidden_identity_key(report.to_dict())
```

Add mutations for plan order, full payload, method binding, input descriptor,
extra records, skipped records, configuration ID, budget, and caller-supplied
completeness.

- [ ] **Step 2: Run the tests and verify RED**

Expected: import failure for `maskimpute_benchmark.fair_comparator_checkpoint`.

- [ ] **Step 3: Implement the direct budget ledger**

Use configuration IDs as keys:

```python
def configuration_budget_key(entry: DirectPlanEntry) -> str:
    return entry.identity.configuration_id


def budget_scope(entry: DirectPlanEntry) -> str:
    identity = entry.identity
    return (
        f"{identity.method.method_id}:{identity.configuration_kind}"
        if identity.method.method_id == "maskimpute"
        else identity.method.method_id
    )
```

`DirectDevelopmentBudget.authorize`, `record`, and `restore` accept this plain
configuration key. `to_dict` uses `configuration_ids` and `consumed_seconds`.
Central replay validates record/entry identity equality before restoring.

- [ ] **Step 4: Implement the closed direct checkpoint schema**

```python
@dataclass(frozen=True, slots=True)
class DirectCheckpointReport:
    schema_version: int
    identity_mode: Literal["direct-v1"]
    authority_revision: str
    plan_snapshot: Mapping[str, object]
    input_descriptors: tuple[PreparedInputDescriptor, ...]
    planned_run_count: int
    status: Literal["running", "completed"]
    evaluation_scope: Literal["reconstruction_only"]
    comparator_selection_status: Literal[
        "complete_terminal_denominator",
        "blocked_incomplete_denominator",
    ]
    selection_complete: bool
    selection_blockers: tuple[str, ...]
    records: tuple[Mapping[str, object], ...]
    budget: Mapping[str, object]
```

The stored JSON contains exactly those fields. `DirectCheckpointStore.load`
accepts `(plan, *, registry, prepared_datasets)`, compares the complete plan and
input descriptors, rejects excess/non-prefix records, derives completeness,
replays the one ledger, and requires exact budget equality. Write and append
derive rather than accept a budget object.

- [ ] **Step 5: Port transaction-intent recovery**

The intent stores the complete entry identity and provisional direct record.
Recovery accepts only the next ordinal and exact plan equality. A durable
`infrastructure_error` record remains durable; only an intent without its
corresponding record may complete or be retried.

- [ ] **Step 6: Run exact Task 7 and adjacent checkpoint suites**

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_fair_comparator_checkpoint.py tests/test_benchmark_runner.py \
  tests/test_prezero_evidence.py tests/test_development_evaluation.py \
  -k 'direct or budget or checkpoint or transaction or infrastructure or comparator_selection' \
  -q -W error -p no:cacheprovider
```

- [ ] **Step 7: Commit**

```bash
git add maskimpute_benchmark/fair_comparator_checkpoint.py maskimpute_benchmark/runner.py \
  tests/test_fair_comparator_checkpoint.py tests/test_benchmark_runner.py \
  tests/test_prezero_evidence.py tests/test_development_evaluation.py
git commit -m "fix: replay direct comparator checkpoints exactly"
```

### Task 6: Integrate direct evidence and enforce the migration boundary

**Files:**

- Modify: `maskimpute_benchmark/development_evaluation.py`
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `maskimpute_benchmark/selection.py`
- Modify: `maskimpute_benchmark/runner.py`
- Modify: `tests/test_development_evaluation.py`
- Modify: `tests/test_downstream_evidence.py`
- Modify: `tests/test_selection_authority.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `docs/superpowers/plans/2026-07-18-fair-comparator-tuning.md`

**Interfaces:**

- Produces: one direct fair-comparator base entry path, a plain full-payload comparator projection, a static migration audit, and a revised continuation contract for Tasks 8–18.
- Leaves legacy `CompetitionPlan`, `ExecutionRequest`, `RawRunResult`, and `CheckpointStore` behavior unchanged for non-fair-comparator callers.

- [ ] **Step 1: Write the failing end-to-end synthetic schema test**

Construct the tracked authority, direct plan, three-record synthetic checkpoint,
and development-evaluation projection without executing an adapter. Assert:

```python
assert projection["comparator_authority"] == {
    "path": "study/comparator_tuning.json",
    "schema_version": 2,
    "authority_revision": "fair-comparator-direct-v1",
}
assert projection["selected_comparators"] == {
    row.method_id: {
        "configuration_id": row.configuration_id,
        "payload": dict(row.payload),
    }
    for row in selected_rows
}
assert not _contains_forbidden_identity_key(projection)
```

- [ ] **Step 2: Run the integration test and verify RED**

Expected: existing consumers require superseded checkpoint and comparator
identity fields.

- [ ] **Step 3: Add schema-specific consumer routing**

Development evaluation and downstream evidence detect `identity_mode` and use
the closed direct schema only for `direct-v1`. They validate the full plan,
records, authority reference, and selected payload map. Mixed fields, caller
selection claims, or a selection receipt before terminal completeness fail.
Legacy routing remains byte-for-byte unchanged.

- [ ] **Step 4: Add a scoped source-and-schema audit**

Create a test that recursively checks tracked comparator sections and all
synthetic direct artifacts for forbidden keys. Parse the AST of
`comparator_tuning.py`, `fair_comparator_plan.py`,
`fair_comparator_execution.py`, and `fair_comparator_checkpoint.py`; reject
imports/calls to content-summary helpers and forbidden dataclass field names.
In shared modules inspect only functions named with `direct` or
`fair_comparator`.

- [ ] **Step 5: Amend the continuation plan**

At the top of `2026-07-18-fair-comparator-tuning.md`, add a governing amendment
that cites the approved direct-identity design and this plan. State that Tasks
8–18 retain scientific requirements but must use the substitution table above,
that old digest-oriented code snippets are superseded, and that each task brief
must include both plan paths.

- [ ] **Step 6: Run focused, adjacent, lint, and compile gates**

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py tests/test_fair_comparator_plan.py \
  tests/test_fair_comparator_execution.py tests/test_fair_comparator_checkpoint.py \
  tests/test_benchmark_runner.py tests/test_development_evaluation.py \
  tests/test_downstream_evidence.py tests/test_selection_authority.py \
  -q -W error -p no:cacheprovider
/tmp/maskimpute-supported/bin/python -m ruff check \
  maskimpute_benchmark/comparator_tuning.py \
  maskimpute_benchmark/fair_comparator_plan.py \
  maskimpute_benchmark/fair_comparator_execution.py \
  maskimpute_benchmark/fair_comparator_checkpoint.py
/tmp/maskimpute-supported/bin/python -m compileall -q maskimpute_benchmark
git diff --check
```

Expected: all focused tests pass, Ruff is clean, compilation succeeds, and the
diff check emits no output.

- [ ] **Step 7: Commit**

```bash
git add maskimpute_benchmark/development_evaluation.py \
  maskimpute_benchmark/downstream_evidence.py maskimpute_benchmark/selection.py \
  maskimpute_benchmark/runner.py tests/test_development_evaluation.py \
  tests/test_downstream_evidence.py tests/test_selection_authority.py \
  tests/test_benchmark_runner.py \
  docs/superpowers/plans/2026-07-18-fair-comparator-tuning.md
git commit -m "refactor: integrate direct comparator evidence"
```

## Post-migration review and continuation

Generate one review package for `869938f..HEAD` and independently review all six
migration tasks for design compliance, ordinary correctness, scientific
fairness, schema closure, and legacy non-regression. Fix every Critical or
Important finding and re-review. Then generate a second package for
`22b02d7..HEAD` and re-review Task 7's exact budget/completeness behavior on the
final direct schema.

After both reviews are clean:

1. append the six migration tasks and final Task 7 acceptance to
   `.superpowers/sdd/progress.md`;
2. resume Task 8 of `2026-07-18-fair-comparator-tuning.md`;
3. include this plan's Global Constraints and substitution table in every Task
   8–18 brief; and
4. extend Task 18's whole-branch gate with the scoped source-and-schema audit.
