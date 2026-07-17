# Development Selection Evidence Workflow

Development selection is an append-only evidence chain. Run each command from
the repository root with the supported locked environments. The commands own
their scientific stages and paths; none accepts an input, output, or revision
override.

## Base stage

After the base development competition, score, calibration, and orthogonal
evidence are complete, run:

```bash
python scripts/build_development_selection_input.py
python scripts/run_development_downstream_evidence.py
python scripts/promote_development_selection_input.py
python scripts/select_development_candidate.py
```

The builder retains schema 2 at
`artifacts/study/development/evaluation/development_selection_input.json`.
Downstream evidence is retained under `evaluation/downstream/`. Promotion
binds both into immutable schema 4 at
`evaluation/development_selection_input-downstream.json`. The selector reads
only that last path and immutably publishes
`evaluation/development_selection_report.json`.

## Conditional v28 stage

Run this sequence only when the exact base report has `trigger=v28`:

```bash
python scripts/run_v28_revision_competition.py
python scripts/build_v28_revision_selection_input.py
python scripts/run_development_downstream_evidence.py
python scripts/promote_development_selection_input.py
python scripts/select_v28_revision_candidate.py
```

The source, downstream archive, complete input, and report use the respective
fixed suffixes `-v28.json`, `downstream-v28/`, `-v28-downstream.json`, and
`-v28.json`. The v28 runner is authorized only by the immutable base schema-4
input and report.

## Conditional v29 stage

Run this sequence only when the exact v28 report has `trigger=v29`:

```bash
python scripts/run_v29_revision_competition.py
python scripts/build_v29_revision_selection_input.py
python scripts/run_development_downstream_evidence.py
python scripts/promote_development_selection_input.py
python scripts/select_v29_revision_candidate.py
```

The v29 artifacts use the corresponding `-v29` suffixes. The v29 runner is
authorized only by the immutable v28 schema-4 input and report.

## Failure and resume rules

- The downstream and promotion commands discover the exact schema-2 base or
  latest schema-3 revision source. An invalid latest source or missing
  downstream archive for that source is an error; neither command falls back
  to an earlier stage.
- Promotion never changes the schema-2/3 source. An identical existing
  schema-4 destination is an idempotent retry; different existing bytes,
  symlinks, non-unique files, and parent-directory swaps are rejected.
- Selection never reads schema 2 or 3. An identical existing report is an
  idempotent retry, while a conflicting report is never overwritten.
- A revision is run only after the preceding fixed report emits its exact
  prespecified trigger. No operator may skip or reorder these steps.
