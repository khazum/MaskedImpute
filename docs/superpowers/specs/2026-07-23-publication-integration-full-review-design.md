# Publication Integration Full Review Design

## Purpose

Review the complete `codex/publication-integration` branch relative to `main`,
correct every reproducible defect within the authorized scope, and establish
whether the resulting software and manuscript infrastructure are internally
consistent. The review assesses the final tree as an integrated system while
also checking the branch-level deletion and migration surface for accidental
loss.

This review does not establish scientific competitiveness or submission
readiness. Those conclusions require the separately authorized real benchmark
runs, frozen analyses, human declarations, and release decisions.

## Scope

The review covers:

- every path added, modified, renamed, or deleted between the merge base with
  `main` and the branch head;
- the `maskimpute` model and training implementation, including numerical
  assumptions, shapes, sparse inputs, calibration, count models, and public API;
- the `maskimpute_benchmark` study protocol, datasets, simulators, method
  adapters, selection authorities, direct-value schemas, checkpointing,
  revisions, publication freeze, final execution, downstream analysis,
  scaling, trajectory analysis, and publication synthesis;
- command-line scripts and their configuration, error handling, resumability,
  deterministic behavior, environment isolation, and failure reporting;
- study JSON contracts and their agreement with code and tests;
- tests for false positives, stale fixtures, bypasses, missing negative cases,
  test-order dependence, and mismatch between test expectations and production
  behavior;
- tracked and ignored repository contents, dependency declarations, submodule
  removals, archived-material moves, generated artifacts, and branch hygiene;
- the Genome Biology Methodology manuscript, bibliography, compilation,
  checklist, venue structure, evidence boundaries, and consistency with the
  executable workflow.

The primary checkout and its unrelated user changes are outside scope. Review
and corrections occur only in `.worktrees/publication-integration`.

## Standing constraints

- Do not perform cyber-related work.
- Do not introduce, extend, or reinstate hashes, checksums, fingerprints, or
  content summaries in the fair-comparator workflow or this review.
- Do not add parent-directory race hardening.
- Preserve legacy outer provenance outside the direct comparator segment.
- Do not execute the real scientific workload. Tests may use bounded synthetic
  fixtures and dry-run or planning modes that cannot produce scientific claims.
- Do not claim that MaskImpute outperforms competitors without completed,
  frozen scientific results.
- Do not invent authors, affiliations, contributions, funding, ethics
  determinations, competing-interest declarations, acknowledgements, licenses,
  repository URLs, archival identifiers, accessions, numerical results,
  figures, or submission authorization.
- Retain legitimate human and scientific blockers as explicit fail-closed
  checklist items.
- Do not modify or clean unrelated changes in the primary checkout.

## Review strategy

The review uses a layered audit rather than treating passing tests as proof of
correctness.

### 1. Establish a reproducible baseline

Record the merge base, branch head, worktree status, change inventory, runtime
versions, test configuration, and previously reported verification commands.
Run fast collection and static checks first. Any baseline failure is diagnosed
before deeper review so later results are not interpreted against a broken
environment.

### 2. Audit the branch structure

Inspect all changed paths with particular attention to the large deletion and
historical-migration surface. Confirm that removals are intentional, that
active imports or documentation do not reference deleted paths, and that
vendored environments, generated products, submodule entries, and archives are
handled consistently.

### 3. Review high-risk software boundaries

Trace the model and publication workflow through these boundaries:

1. study configuration and schema parsing;
2. dataset generation and registry lookup;
3. method registry and adapter execution;
4. smoke qualification and development scheduling;
5. development scores and comparator/candidate selection;
6. revision authorization and execution;
7. publication freeze and direct authority persistence;
8. final, trajectory, downstream, null-DE, and scaling execution;
9. publication synthesis and manuscript-facing claim permissions.

For each boundary, check complete population accounting, type and value
validation, canonical identity binding, deterministic ordering, fail-closed
behavior, replay behavior, exception handling, and prevention of information
leakage from final data into development decisions.

### 4. Review numerical and ML behavior

Check tensor and sparse-matrix shapes, dtype and device transitions, finite
value handling, zero-library cases, loss denominators, mask semantics,
calibration application, count-distribution parameterization, seed propagation,
training/evaluation mode changes, and stable handling of degenerate fixtures.
The review assesses algorithm implementation correctness, not empirical
superiority.

### 5. Challenge tests and contracts

Look for assertions that merely mirror implementation, permissive fixtures that
bypass production constructors, missing mutation cases, unexpected booleans
accepted as integers, incomplete denominator checks, environment or global
state leakage, order-dependent tests, and CLI tests that do not exercise the
real boundary. Add focused negative or metamorphic tests where they can
demonstrate a defect or close a material blind spot.

### 6. Review the publication package

Compile the manuscript from a clean state and inspect logs for unresolved
references, citations, missing assets, and serious layout warnings. Check that
the manuscript uses the intended Genome Biology Methodology structure, that
every methodological statement matches the executable protocol, and that
unavailable evidence remains explicitly unavailable. Remove active review
requirements that contradict the standing no-hashing constraint.

### 7. Correct defects test-first

For each reproducible defect:

1. document the failing invariant and severity;
2. create the smallest regression test that fails for the right reason;
3. run the test and record the failure;
4. implement the minimal correction without unrelated refactoring;
5. run the focused test and adjacent suite;
6. inspect the diff for scope and compatibility;
7. commit a coherent correction.

Documentation-only inaccuracies are corrected against primary repository
evidence and verified with the applicable documentation, compilation, or
hygiene check.

## Severity and disposition

- **Critical:** corrupts or fabricates scientific evidence, leaks final outcomes
  into development selection, silently changes the frozen estimand, or makes
  the workflow materially unsafe to use. Critical findings must be corrected.
- **Important:** produces incorrect outputs, incomplete population accounting,
  non-deterministic authority, invalid publication claims, broken execution, or
  a substantial untested correctness gap. Important findings must be corrected.
- **Minor:** localized maintainability, diagnostic, documentation, or test
  precision issue without incorrect production behavior. Minor findings are
  corrected when they are in scope and low risk; exclusions are recorded
  explicitly.
- **Human/scientific blocker:** requires empirical runs, author judgment,
  institutional determination, third-party identifiers, or publication
  authorization. These are reported but never fabricated or treated as code
  defects.

## Verification

Verification proceeds from focused to whole-branch:

- test collection;
- Ruff lint and exact formatting checks;
- Python compilation;
- focused unit and contract tests for every correction;
- adjacent subsystem suites;
- the complete pytest suite with no unreported exclusions;
- manuscript compilation from a cleaned build directory;
- scans for unresolved citations, references, placeholders, stale active paths,
  and prohibited generated artifacts;
- repository hygiene tests;
- final diff, status, and ignored-artifact inspection.

If the full suite is too long for a single uninterrupted tool call, it may run
in a resumable terminal session. Completion is claimed only from the final exit
status and complete result summary.

## Acceptance criteria

The review is complete when:

- every changed path has been covered by structural, automated, or manual
  review appropriate to its risk;
- no unresolved critical or important defect remains;
- every correction has focused regression evidence;
- static checks, the full test suite, manuscript build, and hygiene checks pass
  at the final branch head;
- the integration worktree is clean;
- no real scientific result or human declaration has been invented;
- remaining minor exclusions and human/scientific blockers are listed with
  their exact impact;
- the final report distinguishes infrastructure correctness from empirical
  competitiveness and Genome Biology submission readiness.

## Deliverables

- committed regression tests and corrections for in-scope defects;
- corrected manuscript or checklists where repository evidence shows an
  inconsistency;
- an evidence-backed review report listing findings by severity, commands run,
  final verification results, excluded items, and unresolved human/scientific
  blockers;
- a clean `codex/publication-integration` worktree ready for the user's chosen
  integration action.
