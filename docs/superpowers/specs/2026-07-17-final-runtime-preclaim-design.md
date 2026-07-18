# Frozen Final Runtime Preclaim Design

**Status:** Approved publication-safety correction

## Purpose

The frozen final runner currently claims or mutates resume state before it proves
that the required external simulator source snapshot and R environment exist and
match tracked authority. Final dataset generation already requires those paths,
but the top-level runner does not accept or pass them. A failed runtime lookup can
therefore leave a claimed round that could never have executed.

The correction makes external runtime validation a fail-closed operational
precondition. It does not add a scientific override, change any method,
configuration, seed, dataset design, or publication denominator, and it does not
record machine-specific paths in scientific authority.

The tracked simulator closure is also sensitive to an inherited user/Codex
executable and loader search path. Supported final entrypoints therefore need a
single sanctioned process-environment boundary before they verify any runtime
source or take the preclaim snapshot.

## Selected design

`run_frozen_final_round` receives two required keyword-only `Path` values:
`simulator_assets_root` and `simulator_r_environment`. After canonical read-only
repository/round and frozen-method validation, it calls the authoritative
simulator-runtime loader with `require_outside_repository=True`. It captures the
path-independent semantic SHA-256 and defensive receipt copy, then closes the
private runtime snapshot deterministically.

Only a successful preclaim permits either new claim issuance or any resume
cleanup, recovery, reconciliation, or journal mutation. Invalid, missing,
symlinked, in-repository, world-writable, or authority-mismatched paths therefore
leave both an unclaimed and an existing running round byte-identical.

The runner passes the same two paths to final panel generation and running-status
preparation. It requires the generated status `runtime_assets_sha256` and
`runtime_assets_receipt` to equal the preclaim values. Running-status preparation
reopens and revalidates the current external authority, closing the interval
between preclaim and method execution. A mismatch aborts before any method
adapter or execution store runs.

`load_prepared_final_panel` accepts optional keyword-only runtime paths. A running
round requires both. A partial pair is always invalid. `allow_evaluated=True`
with neither path preserves the existing frozen-receipt reconstruction and is
the production downstream interface after the one-use final evaluation. Thus a
published result never depends on an original local filesystem path.

The CLI requires `--simulator-assets-root` and
`--simulator-r-environment`. These locate already frozen bytes; all scientific
override flags remain absent.

Both supported final runtime-asset entrypoints use one centralized operational
environment helper. It sets `PATH` exactly to
`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` and removes
`LD_LIBRARY_PATH`. `scripts/run_frozen_final.py` establishes this boundary on
every invocation before the runner can preclaim runtime assets.
`scripts/generate_study_datasets.py` establishes it only for
`--namespace final`, before generation can verify runtime sources. Development
generation does not mutate the caller's environment. Direct library calls
remain fail-closed if invoked under a different environment; the supported CLI
boundary is not a scientific override.

After this code is independently reviewed, the tracked simulator runtime lock
and derived authority must be regenerated once under the sanctioned stable
environment and integrated separately. This implementation branch deliberately
does not edit either authority file.

## Rejected alternatives

- Lazy validation inside dataset generation is rejected because it occurs after
  claim issuance or resume mutation.
- Copying external runtimes into the repository/round is rejected because it
  weakens the outside-repository boundary and makes authority machine-path
  dependent.
- Passing a mutable preloaded runtime object through every simulator layer is
  rejected because it broadens ownership and lifetime semantics unnecessarily;
  the existing authoritative path loader remains the single boundary.
- Duplicating environment literals in both scripts is rejected because the two
  supported entrypoints must not silently diverge.

## Error handling and verification

All runtime/preparation errors become `FinalRunnerContractError` at the runner
boundary. Runtime snapshots close on success and failure. Tests must prove:

- the public API exposes only the two required keyword-only operational paths;
- failed preclaim occurs before new claim issuance and before every resume
  mutation;
- the exact paths reach generation and running-status revalidation;
- generated SHA/receipt drift aborts before method execution;
- partial/missing running path pairs fail;
- evaluated replay with no paths remains byte-identical and read-only; and
- the frozen-final CLI exposes the two operational flags and no scientific
  overrides;
- both final CLI paths expose the exact sanctioned environment to their
  downstream boundary before runtime verification; and
- development dataset generation leaves the caller environment unchanged.
