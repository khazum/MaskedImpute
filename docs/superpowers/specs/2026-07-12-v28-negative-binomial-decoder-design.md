# MaskImpute v28 Negative-Binomial Decoder Design

## Scope

Add a conditional v28 development candidate without changing the frozen v27
public API or its tracked 20-configuration search. The implementation must use
only raw observed counts, external cell IDs, the verified cross-fitted count
score, and the retained calibration artifact. Evaluator truth, labels, and
downstream endpoints remain outside model dispatch.

## Statistical model

The explicit-mask encoder and optimization budget remain unchanged. The v28
decoder emits a simplex of gene fractions for each cell. Its count mean is

`mu[cell, gene] = observed_library_size[cell] * fraction[cell, gene]`.

Artificially held-out observed positives are optimized with the exact
negative-binomial log likelihood using the mean/size parameterization
`variance = mu + mu^2 / theta`. The natural-zero regularizer remains the
external-score-weighted squared prediction penalty on the v27 normalized
scale, obtained as `log1p(fraction * normalization_target)`. Validation uses
only the fixed positive holdout and the NB likelihood.

Gene-wise dispersion is estimated from observed counts and observed library
sizes. Exposure-adjusted method-of-moments contributions are winsorized to
limit individual-cell leverage, then shrunk on the log-dispersion scale toward
the median valid gene estimate. Fixed lower and upper bounds prevent undefined
or numerically degenerate NB size values. Fixed validation positives are
excluded when the training objective constructs this nuisance estimate.

## Integration

`maskimpute/nb_model.py` owns the NB likelihood, dispersion estimate, decoder,
and objective factory. The existing deterministic trainer accepts an optional
objective factory; its default path is byte-for-byte equivalent in behavior to
v27 MSE training. The existing development ablation primitive accepts a fixed
decoder selector and therefore continues to own score-artifact verification,
leave-one-draw-out calibration, output gating, and positive copying.

The benchmark dispatcher accepts v28 only for an authorized candidate-search
configuration whose payload identifies `method_version=v28` and
`decoder=negative_binomial`. Decoder hyperparameters are parsed into an exact
immutable configuration. No v28 symbol is added to the public `maskimpute`
API, and no v28 row is added to the current tracked development-search ledger.

## Failure behavior and auditability

Inputs must be finite integral nonnegative counts. Fractions, means,
dispersions, losses, and outputs must remain finite. Empty objective masks,
nonpositive NB size, invalid configuration fields, unsupported version/decoder
pairs, and zero-information training matrices fail closed. Diagnostics record
the mean parameterization, likelihood, dispersion estimator and bounds,
decoder configuration, training histories, score/calibration receipts, gate,
and output policy.

## Verification

Tests compare the Torch likelihood against `scipy.stats.nbinom.logpmf`, exercise
dispersion robustness and shrinkage, prove the library-offset identity, verify
deterministic training and caller RNG restoration, assert shared calibrated
score/gate/selective-copy behavior, and reject non-authorized decoder payloads.
Focused v27, ablation, method-adapter, runner, and v28 tests plus Ruff form the
completion gate.
