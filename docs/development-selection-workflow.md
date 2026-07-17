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

## Publication freeze

Freeze only after the terminal stage has an exact report with
`trigger=freeze_candidate` and one selected configuration. The valid terminal
chains are:

| Terminal stage | Required report chain |
| --- | --- |
| base | base selects the v27 candidate |
| v28 | base emits `trigger=v28`; v28 selects the v28 candidate |
| v29 | base emits `trigger=v28`; v28 emits `trigger=v29`; v29 selects the v29 candidate |

Prepare the receipt from the repository evidence, then commit it as the sole
change directly on top of the preparation commit:

```bash
python scripts/freeze_publication_round.py prepare
git add -f study/frozen_method.json
git commit -m "freeze publication method"
python scripts/freeze_publication_round.py freeze artifacts/study/round-001
```

The prepare command resolves `base`, `v28`, or `v29` from fixed repository
paths. The CLI deliberately has no stage, input, report, configuration, or
evidence override. It replays every schema-4 selection and revision activation
in the consecutive prefix and binds the exact source inputs, complete inputs,
reports, evaluation manifests, reconstruction checkpoints, orthogonal and
downstream archives, tracked authorities, and closed directory receipts.

For a revision selection, the frozen method retains base execution evidence
for observed and comparator methods and substitutes only MaskImpute's selected
v28 or v29 checkpoint evidence. Both tracked revision specifications and the
scaling and trajectory authorities remain bound in every frozen receipt.

`validate_frozen_method` can validate the committed receipt and tracked
authorities after ignored development artifacts have been removed. Opening a
round is stricter: it re-resolves and replays the raw development package and
requires it to match the committed stage receipt before final evidence starts.

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
- The presence of any known v28 or v29 operational path makes that the newest
  attempted stage. Its entire consecutive prefix must then be complete and
  valid. A partial, malformed, symlinked, or racing newest footprint blocks
  preparation and round freeze; the resolver never falls back to an earlier
  favorable report.
- A `downgrade_claim` is a terminal claim outcome, not permission to freeze an
  earlier method. It blocks publication freeze, as do a selected preceding
  stage, an out-of-order revision, or another revision trigger at the terminal
  stage.
- Frozen-method publication is create-only. Sequential or concurrent retries
  accept only byte-identical canonical content. Different or unsafe existing
  content is preserved and rejected; there is no overwrite or repair path.
