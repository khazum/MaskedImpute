# Development Selection Evidence Workflow

Development selection is an append-only evidence chain. Run each command from
the repository root with the supported locked environments. This document
records the required order; it does not authorize a scientific execution.

## Prespecified release order

The release operator must complete these steps in order:

1. Implement and independently review the separate calibration amendment.
2. Generate the fixed development panel:
   `python scripts/generate_study_datasets.py --namespace dev`.
3. Prepare the development count scores and retained calibration:
   `python scripts/prepare_development_scores.py`.
4. Validate both prepared artifacts and bind them into the tracked development
   authority:
   `python scripts/finalize_development_authority.py`.
5. Commit and independently review `study/development_search.json` together
   with the prepared score and calibration artifacts. Later selection commands
   require this committed clean state.
6. Run the fixed 34-configuration smoke:
   `python scripts/run_comparator_tuning_smoke.py`.
7. Run the complete development denominator:
   `python scripts/run_development_competition.py`.
8. Select one development-only configuration per eligible comparator:
   `python scripts/select_comparator_configurations.py`.
9. Build the fixed base development-selection input:
   `python scripts/build_development_selection_input.py`.
10. Run the fixed base downstream evidence:
    `python scripts/run_development_downstream_evidence.py`.
11. Promote the base input only after that downstream evidence completes:
   `python scripts/promote_development_selection_input.py`.
12. Select the base development candidate:
   `python scripts/select_development_candidate.py`.
13. Complete a revision stage only when the preceding immutable activation
    receipt requests that exact revision. An activated `v28` stage requires the
    entire following sequence:

   ```text
   python scripts/run_v28_revision_competition.py [--environment METHOD=EXECUTABLE ...]
   python scripts/build_v28_revision_selection_input.py
   python scripts/run_development_downstream_evidence.py
   python scripts/promote_development_selection_input.py
   python scripts/select_v28_revision_candidate.py
   ```

    A subsequent exact `v29` activation requires:

    ```text
    python scripts/run_v29_revision_competition.py [--environment METHOD=EXECUTABLE ...]
    python scripts/build_v29_revision_selection_input.py
    python scripts/run_development_downstream_evidence.py
    python scripts/promote_development_selection_input.py
    python scripts/select_v29_revision_candidate.py
    ```

    The revision runners permit only the optional, repeatable
    `--environment METHOD=EXECUTABLE` adapter binding shown by `--help`. The
    builders, downstream runner, promoter, and selectors resolve their fixed
    stage paths; none accepts a stage, input, report, configuration, receipt,
    or output-path override.
14. Complete the fixed external-reference development track before publication
    preparation:

    ```bash
    python scripts/run_external_reference_development.py \
      --environment "d3impute=$D3IMPUTE_PYTHON" \
      --environment "sctsi=$SCTSI_RSCRIPT" \
      --sctsi-library "$SCTSI_LIBRARY"
    ```

    These three values are release-operator locators for the separately fixed
    environments, not scientific overrides.
15. Prepare the publication-round receipt:
   `python scripts/freeze_publication_round.py prepare`.
16. Commit and independently review `study/frozen_method.json` together with
    every authority to which it is bound.
17. Freeze the newly opened round:
    `python scripts/freeze_publication_round.py freeze "$ROUND_DIR"`.
18. Execute the frozen final round:

    ```bash
    python scripts/run_frozen_final.py "$ROUND_DIR" \
      --simulator-assets-root "$SIMULATOR_ASSETS_ROOT" \
      --simulator-r-environment "$SIMULATOR_R_ENVIRONMENT"
    ```

19. Complete the analysis and downstream safety products from the evaluated
    round, with null-DE following the downstream stage:

    ```bash
    python scripts/generate_final_analysis.py "$ROUND_DIR"
    python scripts/run_final_downstream_evidence.py --round-dir "$ROUND_DIR"
    python scripts/run_final_null_de.py --round-dir "$ROUND_DIR"
    ```

`ROUND_DIR`, `SIMULATOR_ASSETS_ROOT`, and `SIMULATOR_R_ENVIRONMENT` are
release-operator paths, not scientific overrides. The operator sets them to
the newly opened frozen round and the separately pinned external simulator
assets and environment, then retains the resulting receipts.

The frozen-final entry point owns the fixed scaling substage and its evaluation
transition. Lower-level state and subsystem commands such as `studyctl.py`,
`generate_study_datasets.py --namespace final`, and `run_scaling_panel.py` do
not replace this guarded release sequence.

## Comparator development evidence

The smoke receipt must cover all 34 registered comparator configurations before
the full development competition is eligible. The complete base denominator is
2,896 scheduled rows: 16 observed-count rows, 48 capacity-matched autoencoder
rows, 1,200 MaskImpute rows, and 1,632 comparator rows. Comparator selection is
development-only: it averages model seeds within technical view, pairs views
within biological draw, and selects one global configuration per eligible
method without reading MaskImpute performance, downstream endpoints, or final
data.

The selected-comparator receipt is fixed at
`artifacts/study/development/evaluation/comparator_selection.json`. The builder
retains schema 2 at
`artifacts/study/development/evaluation/development_selection_input.json`.
Promotion binds the required evaluator-only development archives into immutable
schema 4 at
`artifacts/study/development/evaluation/development_selection_input-downstream.json`.
The selector reads only that last path and immutably publishes
`artifacts/study/development/evaluation/development_selection_report.json`.
Missing required archives or an incomplete selected method binding block
progression rather than reducing the scheduled denominator.

## Conditional revision stages

The v28 runner is authorized only by the immutable base schema-4 input and
report. The v29 runner is authorized only by the immutable completed v28 chain.
Revision source, downstream archive, complete input, and report use their fixed
stage paths; neither runner accepts a free-form rerun request. A revision that
is not activated remains unexecuted.

## Publication freeze

Freeze only after the terminal stage has an exact report with
`trigger=freeze_candidate` and one selected configuration. The valid terminal
chains are:

| Terminal stage | Required report chain |
| --- | --- |
| base | base selects the v27 candidate |
| v28 | base emits `trigger=v28`; v28 selects the v28 candidate |
| v29 | base emits `trigger=v28`; v28 emits `trigger=v29`; v29 selects the v29 candidate |

The prepare command resolves `base`, `v28`, or `v29` from fixed repository
paths. It replays every schema-4 selection and revision activation in the
consecutive prefix and binds the complete source inputs, reports, evaluation
manifests, reconstruction checkpoints, evaluator-only archives, tracked
authorities, and closed directory receipts.

For a revision selection, the frozen method retains base execution evidence
for observed and comparator methods and substitutes only MaskImpute's selected
v28 or v29 checkpoint evidence. Both tracked revision specifications and the
scaling and trajectory authorities remain bound in every frozen receipt.

Opening a round re-resolves and replays the raw development package and requires
it to match the committed stage receipt before final evidence starts. The final
round must contain exactly 1,760 scheduled rows, and the trajectory extension
must contain exactly 44 scheduled rows before analysis or claim rendering.

## Failure and resume rules

- Promotion never changes its source. An identical existing destination is an
  idempotent retry; different existing bytes, symlinks, non-unique files, and
  parent-directory swaps are rejected.
- Selection never reads an incomplete source. An identical existing report is
  an idempotent retry, while a conflicting report is never overwritten.
- A revision is run only after the preceding fixed report emits its exact
  prespecified trigger. No operator may skip or reorder these steps.
- The newest attempted revision must have a complete consecutive prefix. A
  partial or malformed newest footprint blocks preparation and round freeze;
  the resolver never falls back to an earlier favorable report.
- A `downgrade_claim` is a terminal claim outcome, not permission to freeze an
  earlier method. It blocks publication freeze, as do an out-of-order revision
  or another revision trigger at the terminal stage.
- Final reporting retains every scheduled method and a complete reason-coded
  execution-status table. A method without an eligible selected configuration
  is excluded only from numerical estimands that cannot be computed; it cannot
  be silently removed from the denominator or used as the target of an
  availability-dependent performance claim.
