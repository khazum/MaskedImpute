# Cross-scope publication runtime-lock design

## Problem

The publication runtime lock intentionally contains thirteen environments. Eleven
belong to the benchmark process and same-input adapters; `d3impute` and `sctsi`
belong only to the matched-bulk external-reference workflow. The ordinary
`ExecutionEnvironmentRegistry` live-probes the eleven environments it can execute,
but the runtime-lock validator currently requires its live declarations to equal
the full lock ID set. The valid two-entry cross-scope remainder therefore makes
development, scaling, and final registry construction fail closed.

## Selected design

Retain one full publication lock and make the scope split explicit. Runtime-lock
validation receives a closed, disjoint sequence of `lock_only_environment_ids`.
It requires:

```text
lock IDs = live declaration IDs union lock-only IDs
live declaration IDs intersect lock-only IDs = empty
```

Every live declaration is independently probed as before. Lock-only entries are
not probed by the same-input registry, but their canonical inventory hashes are
included in the returned receipt and registry identity. Unknown extras, missing
declared entries, overlap, duplicates, and unsafe IDs fail closed. The default
empty lock-only sequence preserves exact validation for all other callers.
It also preserves the historical two-key validator receipt byte-shape. Only an
explicit nonempty lock-only sequence adds the separate
`lock_only_environment_inventory_sha256s` receipt field; this keeps simulator
runtime semantic receipts and their frozen authority byte-compatible.

The two lock-only IDs are derived from the ready `external_reference_only` rows in
the tracked method registry, rather than duplicated as an operational constant.
Development competition, scaling, and final registry construction pass that exact
derived tuple. They do not add external executables to `executable_paths`, and the
public final CLI remains free of environment-selection options.

The external-reference workflow remains responsible for live D3Impute/scTsI
validation. It already requires their exact executable and scTsI-library locators,
probes each lock entry around execution, revalidates persisted evidence, and is
bound into the publication freeze. The freeze independently requires the complete
thirteen-entry lock and its file hash.

## Scientific denominator

`build_competition_plan` continues to admit only `same_input_required` methods.
For the sixteen development views, the frozen authority yields 1,744 entries:
observed plus ten three-seed same-input comparators, followed by the tracked
MaskImpute search/control configurations. D3Impute and scTsI remain outside this
plan and are evaluated only in their separate matched-bulk workflow.

## Rejected alternative

Live-probing D3Impute and scTsI in development, scaling, and final registries would
require irrelevant executable and R-library locators at every boundary. Supplying
them through the frozen-final CLI would violate its no-environment-override
contract; hard-coded or checkpoint-derived absolute paths would couple final
execution to ignored development state. Scope-aware lock-only binding preserves
the full authority without that operational dependency.

## Verification

Tests must prove exact default behavior, safe/disjoint lock-only validation,
unknown-extra rejection, receipt and full-revalidation retention, registry-derived
forwarding at development/scaling/final boundaries, exact tracked thirteen-ID
parity, and a 1,744-entry plan with no D3Impute/scTsI executions. Existing
external-reference tests must continue proving those two methods execute only in
their dedicated track.
