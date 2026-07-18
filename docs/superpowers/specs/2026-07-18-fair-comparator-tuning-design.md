# Fair comparator tuning and development-denominator design

## Status and scope

This design replaces the default-only comparator portion of the development
competition. It was approved before any fresh development competition or final
seed execution. It does not authorize a scientific run by itself.

The subproject has four responsibilities:

1. give every tunable same-input comparator bounded, development-only tuning;
2. restore BiAEImpute to the exact scheduled comparator denominator;
3. select one comparator configuration per method without consulting
   MaskImpute performance; and
4. freeze the selected payloads unchanged into candidate selection, final
   evaluation, and scaling.

Calibration-policy repair, MaskImpute v28/v29 revisions, publication-asset
export, manuscript results, release licensing, and new real-data panels are
separate subprojects. No competition may run until this design and the separate
calibration amendment are both implemented and reviewed.

## Problem

The current 1,744-entry development plan is not a fair publication comparison.
MaskImpute receives twenty selection-eligible search configurations and five
nonduplicated ablations, while every learned comparator receives only a
`registry-default` entry. That entry hashes method-registry metadata rather than
an adapter configuration. Development dispatch ignores comparator payloads,
and final and scaling execution reconstruct current adapter defaults instead of
using a development-selected, frozen payload.

The selection contract also omits `biaeimpute` from
`required_comparator_ids`, although the method registry declares it
`same_input_required`, the runner schedules it, and the manuscript discusses it.
Consequently, BiAEImpute can disappear from the selection gate without an
authority failure.

Executing the current plan would contradict the study's stated rule that each
method receives the same development-only tuning access, up to twenty
configurations and the common runtime budget. The old 1,744-entry plan and every
checkpoint produced from it remain invalid for publication selection.

## Considered approaches

### Selected: integrated full-panel tuning

Plan and attempt every authorized comparator configuration on all sixteen
development datasets and all three model seeds, subject only to the common
prespecified runtime budget. Select one global configuration per method from
that method's own results. This produces one simple authority chain and the
strongest audit trail.

### Rejected: two-stage half-panel tuning

Tune on eight datasets and rerun selected configurations on the full panel.
This saves 336 executions but leaves only four independent biological tuning
units and duplicates selected-subpanel work. It is less persuasive for a
Genome Biology methods comparison.

### Rejected: nested leave-one-draw-out comparator identities

Select a different comparator configuration for each held-out development
draw. This has attractive bias control but changes comparator identity by fold,
complicates freeze and manuscript interpretation, and does not reduce the
2,896-attempt execution burden.

Default-only evaluation is not an acceptable fallback. It would preserve the
known fairness defect. Conversely, comparator configurations must never be
selected to maximize MaskImpute's apparent advantage.

## Tracked authority

Add `study/comparator_tuning.json` as the sole tracked source of comparator
configuration authority. The canonical document must contain:

- schema and contract identifiers;
- an explicit `development_only` data scope and prohibition on final-data use;
- the exact method order and configuration order;
- one complete adapter payload per configuration, not a patch or registry
  metadata hash;
- a canonical SHA-256 for every configuration payload;
- the selection metrics, collapse order, Pareto and rank rules, tie breakers,
  readiness rules, and runtime budgets;
- the exact scheduled and modern-core comparator sets; and
- the output path and schema for the generated selection receipt.

`study/selection_contract.json` must bind the tuning-authority path and hash and
must move from one overloaded required list to four explicit sets:

```text
scheduled_same_input_ids
required_control_ids
established_comparator_ids
modern_core_ids
```

`scheduled_same_input_ids` is validated structurally as the registry-ordered
list of every noncandidate method whose `execution_scope` is
`same_input_required`:

```text
observed
capacity-matched-ae
alra
magic
dca
scvi
saver
scziva
afmf
biaeimpute
sccr
scsdae
```

It is a scheduling and reporting denominator. Numerical estimands are
necessarily defined only for completed, applicable selected configurations, but
the complete scheduled denominator and every failure are co-reported.
Success-conditioned deletion from tables, figures, captions, or status
summaries is forbidden. Any mismatch between this list and the method registry
fails authority loading.

`required_control_ids` is:

```text
observed
capacity-matched-ae
```

`established_comparator_ids` is:

```text
alra
magic
dca
scvi
saver
```

`modern_core_ids` is:

```text
scziva
afmf
biaeimpute
sccr
```

scSDAE remains scheduled and is included in every numerical comparison when it
has an eligible configuration, but it does not substitute for one of the four
declared modern-core methods.

`study/development_search.json` must bind the comparator-tuning authority and
the revised selection contract. Any change to either file creates a new plan
hash and makes the old checkpoint unusable. The previously documented 1,744-entry
denominator is superseded; cross-scope runtime-lock semantics are unchanged.

The authority has separate `payload_sha256` and raw-file SHA-256 identities.
`payload_sha256` is domain-separated canonical JSON over the authority object
with that field omitted; the raw-file digest binds exact encoded bytes. A
self-referential hash is forbidden. The same rule applies to every generated
receipt that embeds its own payload hash.

## Component boundaries

`maskimpute_benchmark/comparator_tuning.py` owns strict authority loading,
payload decoding, collapse, Pareto/rank selection, readiness assessment, and
canonical selection-receipt validation. It does not execute methods or inspect
candidate performance.

`maskimpute_benchmark/runner.py` owns plan expansion, budget accounting, typed
payload dispatch, and terminal run records. It consumes the tuning module's
validated immutable configuration objects but contains no selection logic.

`maskimpute_benchmark/development_evaluation.py` projects only selected
comparator identities into the existing candidate-selection evidence after
validating the comparator receipt. `maskimpute_benchmark/selection.py` consumes
that projection and cannot reopen comparator tuning.

`maskimpute_benchmark/publication_freeze.py` binds the receipt and exact selected
payload map. `maskimpute_benchmark/final_runner.py` and
`maskimpute_benchmark/scaling.py` consume only that frozen map. A minimal
`scripts/select_comparator_configurations.py` invokes the fixed production path
without scientific overrides. Tests for the new module remain separate from
runner, freeze, final, and scaling integration tests.

## Exact comparator grid

Every payload stores every field accepted by its adapter configuration class.
Fields not named below remain equal to the adapter defaults at the approved
method commit, but they are materialized in JSON rather than filled at runtime.
Unknown fields, missing fields, nonfinite numbers, bool-as-int values, and
invalid enum or range values fail closed. JSON arrays are converted to tuples
only where the typed adapter contract requires a tuple, such as DCA
`hidden_size`.

| Method | Configuration IDs in execution order | Ordered varying values |
|---|---|---|
| ALRA | `alra-default` | `k=0` |
| MAGIC | `magic-t03`, `magic-t01`, `magic-t05`, `magic-t07` | `diffusion_time`: `3`, `1`, `5`, `7` |
| DCA | `dca-h64-32-64`, `dca-h32-16-32`, `dca-h32-32`, `dca-h64-64` | `hidden_size`: `(64,32,64)`, `(32,16,32)`, `(32,32)`, `(64,64)` |
| scVI | `scvi-z10`, `scvi-z05`, `scvi-z20`, `scvi-z30` | `n_latent`: `10`, `5`, `20`, `30` |
| SAVER | `saver-default` | automatic empirical-Bayes default |
| scZiva | `scziva-tau-0p001`, `scziva-tau-0p0001`, `scziva-tau-0p01`, `scziva-tau-0p05` | `tau`: `0.001`, `0.0001`, `0.01`, `0.05` |
| afMF | `afmf-sigma-3`, `afmf-sigma-1`, `afmf-sigma-2`, `afmf-sigma-4` | `sigma`: `3`, `1`, `2`, `4` |
| BiAEImpute | `biaeimpute-z128`, `biaeimpute-z32`, `biaeimpute-z64`, `biaeimpute-z256` | `latent_size`: `128`, `32`, `64`, `256` |
| scCR | `sccr-k15`, `sccr-k05`, `sccr-k10`, `sccr-k30` | `neighbors`: `15`, `5`, `10`, `30` |
| scSDAE | `scsdae-zero-1`, `scsdae-zero-0p25`, `scsdae-zero-0p5`, `scsdae-zero-0p75` | `zero_loss_weight`: `1.0`, `0.25`, `0.5`, `0.75` |

This is thirty-four configurations: eight four-point grids plus two automatic
defaults. ALRA remains automatic because `k=0` invokes upstream spectral rank
estimation. SAVER remains automatic because its empirical-Bayes parameters and
size factors are estimated from the input; `ncores`, `estimates_only`, and
`do_fast` are operational or output-mode choices rather than defensible
truth-guided tuning axes.

Each tuned method varies one scientifically interpretable axis around its
upstream default: smoothing time for MAGIC, architecture width for DCA, latent
capacity for scVI and BiAEImpute, zero threshold for scZiva, rank threshold for
afMF, neighborhood size for scCR, and zero-loss emphasis for scSDAE. Holding all
other fields fixed prevents a combinatorial search from giving one comparator
more development adaptivity than another.

The first configuration listed for every method is the upstream default. It is
not given a metric advantage; it is used only as a late deterministic tie
breaker. Exactly one configuration per method has
`is_upstream_default=true`. Its complete payload must equal a newly constructed
adapter dataclass default at the bound implementation commit. The defaults are
MAGIC `magic-t03`, DCA `dca-h64-32-64`, scVI `scvi-z10`, scZiva
`scziva-tau-0p001`, afMF `afmf-sigma-3`, BiAEImpute `biaeimpute-z128`, scCR
`sccr-k15`, and scSDAE `scsdae-zero-1`, plus the sole ALRA and SAVER entries.

Execution uses the fixed configuration order and attempts the complete
16-dataset by 3-seed block for the default before the remaining blocks. This
implements the already stated "published default is the starting point" rule.
Remaining blocks are attempted in their listed order, which is fixed before
results exist and cannot be reordered after a failure. Selection is nevertheless
forbidden after a budget-truncated grid, so execution order cannot determine the
set of configurations eligible for selection.

## Plan and budget semantics

Add `comparator_tuning` to `AuthorizedConfiguration.kind`. Each of the
thirty-four comparator configurations expands over sixteen development datasets
and model seeds 42, 43, and 44. The plan is:

| Component | Plan entries |
|---|---:|
| Observed input | 16 |
| Capacity-matched autoencoder | 48 |
| Twenty MaskImpute search configurations plus five nonduplicated ablations | 1,200 |
| Thirty-four comparator configurations | 1,632 |
| **Total** | **2,896** |

Every attempt is planned before execution. Budget exhaustion, upstream failure,
timeout, unavailability, and resource excess produce reason-coded entries; they
do not remove entries from the denominator.

The existing common limits remain authoritative per method across all datasets,
seeds, and configurations:

- at most twenty distinct configurations;
- at most eight cumulative GPU-hours for GPU methods;
- at most twenty-four cumulative wall-clock hours for CPU-only methods; and
- the existing six-hour per-run timeout and 48 GiB RAM/14 GiB GPU caps.

All four configurations of a method share one budget ledger. A non-
infrastructure attempt consumes the configuration and time budgets. Before the
scientific competition, a fixed nonstudy smoke fixture must establish that the
closed grid is operationally feasible under the tracked cap. Smoke outputs are
discarded, evaluator truth and performance metrics are never computed, and only
runtime/resource feasibility may inform a pre-study amendment. If the grid is
not feasible, the budget or grid must be amended and reviewed before any study
attempt.

`failed`, `timeout`, `resource_exceeded`, and `unavailable` are intrinsic
terminal outcomes. `budget_exhausted`, `blocked_authority`, and a persisted
`infrastructure_error` are non-scientific incomplete outcomes and always block
comparator selection and publication readiness. A method can be selected only
after every authorized configuration entry is either completed or has an
intrinsic terminal outcome. Selection from a budget-truncated grid is
forbidden.

A persisted `infrastructure_error` is terminal for that checkpoint and consumes
no scientific budget, but it cannot be retried in place. Only an interrupted
transaction that never durably published a record is idempotently resumed.
Repairing infrastructure or adapter code, changing a budget, or changing an
authority requires a new development authority and a fresh checkpoint; it
cannot rewrite the old evidence. Intrinsic terminal outcomes cannot be retried
selectively after inspecting candidate results.

Before creating a checkpoint, development preflight must compute a fail-closed
bound from the exact plan and retained dataset shapes. For each executable entry,
the bound includes two dense little-endian float64 matrices, bounded stdout and
stderr, and the executor receipt. Each MaskImpute entry additionally includes the
existing zlib compression bound for one float64 `p_pre_zero` matrix. The bound
also includes every planned JSON record, one maximum-size checkpoint, and one
shared 1 GiB reserve. Existing constants for log, receipt, record, checkpoint,
and zlib ceilings are reused; empirical compression ratios are forbidden. The
expected retained footprint is approximately 20--27 GiB. Preflight fails before
the first scientific write and must not delete or relocate evidence
automatically.

## Dispatch and method identity

The runner must no longer map every comparator to `registry-default`. Each
comparator run carries:

```text
method_id
configuration_id
configuration_kind=comparator_tuning
configuration_payload
configuration_payload_sha256
tuning_authority_file_sha256
tuning_authority_payload_sha256
```

Stable identity fields are separate and closed:

```text
registry_method_sha256
configuration_payload_sha256
tuning_authority_file_sha256
tuning_authority_payload_sha256
source_authority_sha256
runtime_lock_sha256
environment_registry_sha256
configuration_method_identity_sha256
```

Every authorized configuration has a pre-selection
`configuration_method_identity_sha256`, a domain-separated canonical hash over
the preceding stable fields. Per-run source/environment receipt hashes are
stored separately and cannot redefine this configuration identity. Two
configurations of the same method therefore cannot collide in the plan,
checkpoint, or selection records. The comparator-selection receipt sets
`selected_method_identity_sha256` equal to the selected configuration's
pre-existing identity; it does not derive a new post hoc method identity.

The dispatcher decodes the closed payload into the exact adapter dataclass and
passes that instance to the adapter. It may not reconstruct defaults, accept a
partial override, or allow a caller-supplied configuration. The decoded payload
is reserialized and rehashed before and after execution so that normalization
cannot change its meaning.

## Comparator selection

Generate
`artifacts/study/development/evaluation/comparator_selection.json` only from a
validated 2,896-entry checkpoint. The generator is create-only, atomic,
restart-safe, and idempotent for identical bytes. It accepts no method, metric,
threshold, input-directory, or output-directory overrides.

Selection is performed separately for each method. Candidate values and
configuration identities are excluded from the comparator-selection projection
and cannot affect selected IDs, eligibility, Pareto sets, ranks, or selection
tuples. The full checkpoint envelope is still validated and hash-bound, so a
candidate-byte change can change the receipt binding hash without changing any
comparator decision. Downstream endpoints, external-reference results, and final
data are unavailable to the selector.

### Eligibility and collapse

A comparator configuration is eligible only when all 16 datasets by 3 seeds
have completed successfully and all applicable values exist for these six
lower-is-better metrics:

```text
mse
mse_dropout
gnrmse
mse_pre_dropout_zero
corr_err
mse_non_dropout_nonzero
```

`mse_pre_dropout_zero` is required only for the four SymSim draw-by-view
datasets. The collapse order is fixed:

1. arithmetic mean over the three model seeds within each dataset view;
2. arithmetic mean over the paired moderate and severe views within each
   biological draw; and
3. retain eight biological units for the five panel-wide metrics and two SymSim
   biological units for `mse_pre_dropout_zero`.

Technical views and model seeds never count as independent replicates.

### Pareto and deterministic rank rule

For every eligible configuration, compute the median biological-unit value for
each applicable metric. Pareto dominance is evaluated over all six metric
medians among all eligible configurations. Remove a configuration only when
another configuration of the same method is no worse on every metric median and
strictly better on at least one.

For the remaining Pareto set only, calculate average ranks among configurations
within each biological unit, then take the median unit rank for each metric.
With at most four configurations, unit ranks are half-integers and the median
over two or eight biological units is a quarter-integer. Store each metric rank
exactly as `metric_rank_quarters = 4 * rank`; floating rank serialization is
forbidden. Select the configuration minimizing this deterministic integer
tuple:

```text
(
  maximum_metric_rank_quarters,
  sum_metric_rank_quarters,
  mse_rank_quarters,
  mse_dropout_rank_quarters,
  gnrmse_rank_quarters,
  mse_pre_dropout_zero_rank_quarters,
  corr_err_rank_quarters,
  mse_non_dropout_nonzero_rank_quarters,
  upstream_default_penalty,
  configuration_id
)
```

The maximum and sum cover exactly the six per-metric quarter-ranks; the sum is
equivalent to comparing the mean because the denominator is always six. Ranks
are ascending because all metrics are lower-is-better. Average ranks handle
exact ties. `upstream_default_penalty` is zero only for the declared upstream
default and one otherwise. It is reached only after all performance rank
components tie. `configuration_id` is the final byte-stable tie breaker. This
rank aggregate performs deterministic hyperparameter selection within one
method; it is not a combined scientific efficacy score. One configuration is
selected globally per method; per-dataset, per-mechanism, per-draw, and
final-data-specific selections are forbidden.

### Selection receipt

The canonical receipt binds:

- the tuning-authority file and payload hashes;
- selection-contract and methods-registry hashes;
- checkpoint file, payload, and plan hashes;
- dataset, seed, QC-policy, score, and calibration bindings inherited from the
  checkpoint;
- every configuration's complete terminal-status counts and reason histogram;
- eligibility, collapsed metric values, Pareto membership, metric ranks, and
  deterministic selection tuple;
- the exact selected payload and hash for every selectable method;
- the selected configuration's pre-existing
  `configuration_method_identity_sha256`, repeated as
  `selected_method_identity_sha256`, while every nonselected configuration
  identity remains in the receipt;
- for a method with no selected configuration, a null selected payload and a
  domain-separated `nonexecution_identity_sha256` over the method-registry entry,
  tuning authority, selection receipt namespace, and complete failure
  denominator;
- the readiness assessment described below; and
- its own canonical payload hash.

Nonselected configurations remain fully represented in this receipt and in the
checkpoint. Only selected configurations are projected into candidate-selection
records as comparator identities.

## Denominator and availability policy

Scheduling, numerical comparison, and publication readiness are distinct:

1. **Scheduled denominator:** every ID in `scheduled_same_input_ids` and every
   authorized configuration must have its exact planned entries and terminal
   statuses. No method may disappear because it failed.
2. **Numerical denominator:** numerical values exist only for completed,
   applicable cells of eligible selected configurations. Every selected method
   remains displayed even when a later cell fails, and a missing value is never
   fabricated. A successfully selected comparator cannot be excluded from a
   metric, figure, table, caption, status summary, or claim.
3. **Publication readiness:** every `required_control_ids` control must be
   complete; every `established_comparator_ids` method must be selectable; and
   at least three of the four `modern_core_ids` must be selectable.
   Infrastructure errors or blocked-authority entries always block readiness. A
   modern-core method with no eligible configuration may be reason-coded
   report-only unavailable only after every planned entry is completed or has
   an intrinsic terminal outcome. Mixed completed/intrinsic-failure patterns
   with no globally eligible configuration are permitted only for modern-core
   methods and scSDAE and retain their complete failure pattern.

BiAEImpute is therefore always scheduled, audited, and counted in the modern
readiness denominator. If it is the sole unavailable modern method and the other
three complete, the study may proceed, but the paper must report its full
failure denominator and may not claim superiority over it. If fewer than three
modern-core methods complete, freeze is prohibited. scSDAE is always attempted
and reported; if selectable, it is included in all applicable numerical claims.

This policy prevents a broken third-party method from erasing the complete
study, while preventing silent success-conditioned comparator omission.

## Freeze, revision, final, trajectory, and scaling propagation

Comparator readiness must pass before any candidate is assessed. Candidate rank
and Pareto gates include observed, capacity-matched AE, every selected
established comparator, every selected modern-core comparator, and selected
scSDAE when available. Learned-comparator gates use all selected learned
comparator/control identities. Nonselected configurations never enter candidate
gates. Unavailable methods remain in scheduled and status denominators and
preclude superiority claims against them.

Candidate selection binds the comparator-selection receipt and compares each
candidate only with the one selected configuration per selectable method. It
may not treat nonselected configurations as separate comparators or use them to
increase the candidate's chance of winning.

Publication freeze stores the exact selected configuration ID, full payload,
payload hash, tuning-authority hash, and comparator-selection receipt hash for
every selectable comparator. It also stores the complete scheduled and failure
denominators. Any missing selected payload, default reconstruction, authority
drift, or receipt mismatch fails freeze.

Final, trajectory, and scaling runners decode only those frozen payloads. They
must not load current adapter defaults or rerun comparator tuning. MaskImpute
v28/v29 execution plans contain only newly authorized MaskImpute candidate
configurations. They reuse, without re-execution, the base checkpoint's frozen
observed, capacity-control, and selected-comparator records. Revision selection
binds the base comparator-selection receipt and exact selected-output records;
comparator and control reruns are forbidden. A candidate revision does not
reopen comparator tuning. Final data can validate comparative performance but
can never change a selected configuration.

The structural final denominator remains exactly 1,760 entries. When every
scheduled comparator is selectable, 1,480 are executable and 280 are
prespecified, reason-coded nonexecutions. Each unavailable stochastic
same-input method reclassifies its existing 120 dataset-by-seed entries from
executable to reason-coded nonexecution. It must not collapse them to forty
seed-null entries. The frozen `nonexecution_identity_sha256` replaces a selected
configuration identity on those entries. The 1,760-entry denominator is
unchanged; only the executable/nonexecution split changes. The
comparator-tuning expansion changes development cardinality only.

The same rule applies to every frozen evaluation plan. The all-selectable
supplementary trajectory plan has forty-four entries. An unavailable stochastic
same-input method reclassifies its three seeded trajectory entries without
changing that total; it must not replace them with one seed-null entry.
Nonexecution changes action and identity only. It never changes a prespecified
stochastic seed denominator or plan cardinality.

## Failure handling and auditability

- A malformed tuning authority, payload, configuration hash, plan count, or
  registry-derived denominator fails before execution.
- A partial old checkpoint cannot be resumed because its authority and plan
  hashes differ.
- Changing the tuning authority or its descendants invalidates the comparator
  selection receipt, development selection inputs and reports, revision
  activations, frozen method, any opened final round, final/trajectory/scaling
  plans, evaluation receipts, and publication assets. No descendant may be
  coherently rehashed in place.
- Selection fails closed on missing entries, duplicated identities, unexpected
  metrics, nonfinite values, inconsistent seed/view collapse, or status changes.
- All output publication uses existing no-symlink, no-special-file,
  create-only, atomic, and immediate-revalidation conventions.
- No adapter output is clipped or otherwise repaired to make it numerically
  valid. Native invalid values become reason-coded failures.
- Execution and selection logs must not expose absolute private paths or raw
  stderr; stable hashes and reason codes are retained instead.
- The manuscript execution-status table must include every scheduled method,
  including unavailable and resource-exceeded methods.

## Verification requirements

Tests must prove at least the following:

1. The tracked authority contains exactly thirty-four configurations in the
   fixed method/configuration order, with complete payloads, correct file and
   payload hashes, and exactly one implementation-matching upstream default per
   method.
2. The exact development plan contains 2,896 unique entries with 1,632
   comparator attempts, includes BiAEImpute, and excludes
   external-reference-only methods.
3. Every grid value reaches the correct adapter dataclass and native adapter;
   defaults, partial payloads, unknown fields, type coercion, and payload drift
   are rejected.
4. Configuration identities cannot collide within one method.
5. Per-method configuration/time budgets restore exactly from checkpoints;
   budget exhaustion, blocked authority, and persisted infrastructure errors
   block selection rather than choosing an execution-order-favored subset.
6. Seed-to-view-to-draw collapse, six-metric applicability, all-eligible Pareto
   filtering, Pareto-only average ties, quarter-rank integer encoding, and the
   complete deterministic selection tuple match golden examples.
7. Changing any MaskImpute or final value cannot change comparator selection
   decisions, although a full-checkpoint receipt binding necessarily changes.
8. Nonselected configurations remain in the audit receipt but never appear as
   comparator identities in candidate selection.
9. The scheduled list is derived exactly from the method registry; omitting or
   reclassifying BiAEImpute fails authority validation.
10. Every required control, every established comparator, and at least three of
    four modern-core methods are required for readiness; mixed intrinsic
    failures remain reportable, while non-scientific incomplete outcomes block.
11. Freeze binds exact selected payloads, revision execution reuses base
    comparator records, and final, trajectory, and scaling paths use frozen
    identities without reconstructing defaults.
12. Tuning-authority, checkpoint, selected-payload, freeze, revision, final,
    trajectory, scaling, evaluation, or publication-asset tampering fails before
    a scientific receipt can be published.
13. Existing runtime-lock, source, score, calibration, QC, and simulator
    authorities remain bound.
14. Storage preflight fails before writing when the computed reserve is
    insufficient.
15. The final plan always has 1,760 entries; one unavailable stochastic method
    reclassifies exactly 120 seeded entries and binds its nonexecution identity.
16. The trajectory plan always has forty-four entries; one unavailable
    stochastic method reclassifies exactly three seeded entries rather than
    collapsing them to one seed-null entry.
17. Candidate gates use the exact ready comparison population, and v28/v29
    plans cannot rerun controls or comparators.
18. Static and repository-hygiene checks reject caches, temporary outputs, and
    unintended scientific artifacts.

## Completion boundary

Implementation is complete only when the authority, dispatcher, budget,
selection receipt, availability gate, freeze propagation, revision reuse,
final/trajectory/scaling propagation, tests, and documentation are integrated
and independently reviewed. That completion authorizes the subsequent
calibration-policy subproject; it does not authorize running the development
competition. Fresh scientific execution remains blocked until the calibration
contradiction is separately resolved and the combined authority is reviewed.
