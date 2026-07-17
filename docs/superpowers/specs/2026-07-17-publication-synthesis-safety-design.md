# Publication Synthesis Safety Design

**Status:** Approved implementation companion to the Genome Biology study
design and frozen trajectory/final null-DE design

## Purpose

Produce one read-only, canonical claim-permission synthesis for an evaluated
frozen final round. The synthesis decides only which prespecified claim classes
the evidence permits. It does not write a scientific artifact, select favorable
results, create a new threshold, or turn supplementary trajectory evidence into
an efficacy or safety gate.

## Authority boundary

The public production entry point accepts only a repository path and round
path. It derives the fixed receipt-namespaced downstream and final null-DE
locations; callers cannot supply an evidence directory, source kind, threshold,
candidate, comparator, or endpoint override.

Before synthesis, it independently replays the existing authoritative loaders:

1. `generate_final_analysis` for the evaluated primary final records and their
   receipt-embedded trajectory binding;
2. `validate_frozen_method` for the exact freeze-validity gate bundle and its
   primary-report payload binding;
3. `build_final_downstream_evidence_plan`,
   `load_downstream_evidence_plan`, and
   `load_downstream_evidence_manifest` for exact final downstream evidence;
4. `build_final_null_de_plan` and `load_final_null_de_manifest` for the complete
   recomputed final null-DE archive; and
5. `load_publication_scaling_evidence` for the evaluated scaling checkpoint and
   retained result bytes.

The synthesis accepts only the production final source kind with evidence scope
`all`. It requires the persisted downstream plan to equal the independently
rebuilt plan, the null-DE source plan to equal that downstream plan, and the
round ID, evaluation-receipt hash, result-manifest hash, final plan hash,
execution-manifest hashes, scaling hashes, and denominator counts to agree
across all loaded authorities. It also requires the final analysis to carry the
receipt-validated `trajectory_evidence_sha256` binding. A missing or changed
archive, mismatched binding, coordinated locally rehashed replacement, test
source kind, or caller-selected alias aborts synthesis.

The frozen development selection gate bundle must still be present and valid as
a freeze-validity prerequisite. Its numerical ranks, effects, and safety values
are development evidence and never satisfy a final publication gate.
The synthesis reports only the selected identity and the three passed freeze
flags; it does not copy development numerical assessment values into claim
evidence.

## Frozen final reconstruction gate

`final_analysis.py` adds a `reconstruction_claim_gate` while its authoritative
normalized final records are in memory. The gate obtains the exact three rank
metrics and four Pareto dimensions from the frozen metric-direction contract,
which is already checked against the immutable selection implementation at the
study method commit. The required same-input comparator denominator comes from
the frozen selection contract.

For every method, metric, mechanism, biological draw, and technical view, model
seeds are averaged first. Paired technical views are then averaged within the
biological draw. A value enters the claim gate only when every planned,
applicable raw row for that method and dataset view is successful. Exact
truth-kind structural unavailability is allowed only where the frozen metric
contract makes that metric inapplicable. Candidate or required-comparator
failure, absence, mismatched view coverage, or incomplete applicability makes
the reconstruction gate `unavailable`; it is never success-conditioned away.

Within each complete biological draw, the candidate receives the tie-aware
average rank used by the frozen selection policy against every required
same-input comparator. The median biological-draw rank must be at most 2 for
`mse`, `mse_dropout`, and `gnrmse`. The final Pareto gate uses exactly
`mse_dropout`, `mse_pre_dropout_zero`, `corr_err`, and
`mse_non_dropout_nonzero`, with pre-dropout-zero restricted to its exact-truth
domain. Complete evidence that exceeds a rank threshold or is Pareto dominated
is `failed`; complete evidence satisfying every rank and Pareto condition is
`passed`.

## Final null-DE safety gate

The synthesis reads only records returned by `load_final_null_de_manifest` and
requires one exact row for every source-plan entry. Candidate and observed rows
must cover identical primary-final dataset views. Within each dataset view,
candidate model seeds are averaged first; paired technical views are then
averaged within each `(mechanism, biological_id)` independent draw. Observed
counts follow the same view collapse.

The gate is `unavailable` if any candidate or observed row is noncompleted,
nonfinite, outside `[0, 1]`, duplicated, missing, or has a mismatched dataset,
draw, or view denominator. Over the complete collapsed denominator it is
`passed` only when both conditions hold:

- the maximum MaskImpute draw-level FPR is at most `0.06`; and
- the maximum paired draw-level difference, MaskImpute minus observed counts,
  is at most `0.01`.

Complete evidence violating either limit is `failed`. Threshold comparisons
are inclusive.

## Other evidence and claim permissions

Final downstream and scaling evidence are mandatory replayed prerequisites,
but this task invents no numerical threshold for either. Their numerical gate
status is `not_prespecified`, while their endpoint/run statuses and reason
counts remain in the synthesis denominator report. A later numerical gate can
be added only with a separately frozen authority.

Trajectory evidence is always labeled `descriptive_only`. It has no pass/fail
gate, cannot improve reconstruction or safety status, and cannot rescue a
failed or unavailable claim gate.

Every scientific gate uses exactly `passed`, `failed`, or `unavailable`.
Binding failures raise `PublicationSynthesisError` instead of being converted
into scientific unavailability. The overall competitive gate is `failed` if a
complete prespecified scientific gate fails, otherwise `unavailable` if any
prespecified gate is unavailable, and otherwise `passed`. The word
`competitive` is permitted only in the last case and only after every evidence
prerequisite validates.

Superiority is endpoint-specific and can use only final primary pairwise
evidence against the strongest applicable required comparator. Permission
requires a complete final comparison, favorable 95% interval excluding zero,
and successful Holm adjustment for the prespecified primary-metric family with
adjusted value at most `0.05`. Missing multiplicity evidence, a non-strongest
comparison, a zero-crossing interval, or a non-primary endpoint never permits
superiority language. Trajectory evidence can never enter this condition.
For each lower-is-better primary endpoint, the strongest applicable comparator
is the complete required comparator with the lowest median over frozen-final
draw-collapsed values; method ID is the deterministic tie-break and all tied
method IDs are disclosed. Superiority is `unavailable` when no comparator has a
complete denominator or the selected comparator lacks its exact pairwise/Holm
result. A pairwise result is complete only when its independent-draw count
equals that comparator's complete draw summary, it has both prespecified views
per draw, it cites the exact frozen direction source, and it has no duplicate,
zero-denominator, nonrepresentable-effect, or bootstrap exclusions. Rows from
prespecified structurally inapplicable truth domains do not invalidate an
otherwise exact applicable denominator. Holm evidence must cover the complete
protocol primary-metric family, not a favorable subset.

## Output and failure behavior

The returned schema contains exact evidence bindings, prerequisite summaries,
the reconstruction and null-DE gates, the descriptive trajectory role,
downstream/scaling `not_prespecified` statuses, competitive permission, named
endpoint superiority permissions, and a canonical self-hash. It contains no
manuscript prose. Inputs are revalidated after computation where the underlying
loader provides that protection; malformed evidence raises and no partial
synthesis is returned.

## Files and verification

- Modify `maskimpute_benchmark/final_analysis.py` and
  `tests/test_final_analysis.py` for the authoritative final reconstruction
  claim gate.
- Create `maskimpute_benchmark/publication_synthesis.py` and
  `tests/test_publication_synthesis.py` for exact evidence replay, null-DE
  aggregation, and claim permissions.
- Run focused warning-strict tests first, then the relevant final-analysis,
  downstream, null-DE, scaling, and trajectory suites, Ruff, byte compilation,
  and `git diff --check`.
