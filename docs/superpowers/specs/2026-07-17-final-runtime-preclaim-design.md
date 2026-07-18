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

## Rejected alternatives

- Lazy validation inside dataset generation is rejected because it occurs after
  claim issuance or resume mutation.
- Copying external runtimes into the repository/round is rejected because it
  weakens the outside-repository boundary and makes authority machine-path
  dependent.
- Passing a mutable preloaded runtime object through every simulator layer is
  rejected because it broadens ownership and lifetime semantics unnecessarily;
  the existing authoritative path loader remains the single boundary.

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
- the CLI exposes the two operational flags and no scientific overrides.

