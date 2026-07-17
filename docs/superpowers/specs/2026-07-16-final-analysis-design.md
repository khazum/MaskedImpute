# Frozen Final Analysis Design

## Purpose

Produce a publication-grade, machine-readable analysis of a completed frozen
final benchmark without changing the evaluated round or creating scientific
results that are not present in its immutable evidence.  The report is an
analysis artifact, not a manuscript claim: unsupported estimands remain
explicitly unavailable.

## Authority boundary

The generator accepts only a repository and an evaluated round directory.  It
validates the study lifecycle chain, evaluation receipt, declared result-file
allowlist, frozen repository, execution manifest, and every referenced record
before using a metric.  The evaluation receipt must bind the execution
manifest's raw-file hash, canonical payload hash, final plan hash, and ordered
canonical record hashes.  Records must be canonical JSON and unique regular
files beneath the declared execution directory.

The command-line interface has one positional round locator and no scientific
override flags.  It writes no files into the immutable round.  Canonical JSON,
including a self-hash, is emitted to standard output so callers can archive it
outside the frozen evidence boundary.

## Fixed analysis policy

- The candidate method is read from the frozen selection contract.
- The declared metric family is the protocol's ordered `primary_metrics` set.
- Hierarchical paired bootstraps use 10,000 replicates, seed 20,260,712, and a
  two-sided 95% percentile interval.
- Methods are paired at dataset/technical-view level.  Model seeds and views
  are repeated measurements.  Biological simulation draws, stratified by
  mechanism, are the only independent inference units.
- Deterministic methods' JSON `null` seed is represented only inside the
  existing statistics adapter by a declared singleton sentinel.  This changes
  no metric value and cannot increase the independent sample size.
- A completed metric row is normalized to analytic status `ok`.  Failed,
  timeout, resource-exceeded, and unavailable rows retain distinct statuses
  and reason codes.
- Holm adjustment is performed separately for each candidate-versus-comparator
  comparison over available hypotheses in the declared primary metric family.
- Pareto membership is computed only when frozen protocol/contract authority
  explicitly declares lower-is-better directions for eligible reconstruction
  metrics.  Missing or non-lower directions produce an unavailable Pareto
  section rather than a guessed direction.

## Report sections

The canonical report contains:

1. Input and policy bindings, including all relevant receipt, manifest, and
   record hashes.
2. Run and metric denominators, terminal/analytic status counts, and unavailable
   reason counts, without success-conditioned deletion.
3. Per-method/per-metric median and interquartile range after first collapsing
   seeds within a dataset/view and then views within a biological draw.
4. Candidate-versus-comparator paired effects: median relative effect, 95%
   interval, probability of improvement, biological-draw win/tie/loss counts,
   paired denominator diagnostics, bootstrap count/checksum, and exclusions.
5. Raw and Holm-adjusted two-sided bootstrap sign probabilities within the
   declared metric family.
6. Seed, biological-draw, and technical-view variance components with their
   identifiable denominators and explicit reasons for unavailable components.
7. Pareto non-domination across authority-declared lower-is-better core metrics,
   or a structured unavailable reason.

The full bootstrap arrays are not copied into the report.  Their deterministic
replicate count and byte-level checksum preserve a compact recomputation
binding.

## Failure behavior

Malformed, noncanonical, incomplete, relocated, undeclared, hash-mismatched, or
concurrently changed evidence aborts generation.  Scientific absence does not
abort generation: missing pairs, nonrepresentable variance, absent direction
authority, and unavailable metrics are retained as explicit structured
unavailability.  No cells or genes are ever used as inferential replicates.

## Files

- `maskimpute_benchmark/final_analysis.py`: evidence validation, normalization,
  fixed-policy analysis, and canonical report generation.
- `scripts/generate_final_analysis.py`: one-locator CLI that prints the report.
- `tests/test_final_analysis.py`: synthetic frozen-evidence and pure-analysis
  contract tests.

