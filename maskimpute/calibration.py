"""Leakage-safe development calibration for count-derived pre-zero scores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


PROBABILITY_CLIP = 1e-12
CALIBRATOR_ORDER = ("identity", "logistic", "beta", "isotonic")
_MECHANISM = re.compile(r"[a-z][a-z0-9_]*")
_BIOLOGICAL_ID = re.compile(r"draw-(?:0[1-9]|[1-9][0-9])")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _numeric_array(value: object, name: str, *, ndim: int | None) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} must not be a masked array")
    coerced = np.asanyarray(value)
    if np.ma.isMaskedArray(coerced):
        raise TypeError(f"{name} must not be a masked array")
    if coerced.dtype.metadata is not None:
        raise TypeError(f"{name} dtype metadata is not supported")
    if coerced.dtype.kind not in "iuf" or coerced.dtype.kind == "b":
        raise TypeError(f"{name} must contain real numeric values")
    if ndim is not None and coerced.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    result = np.array(coerced, dtype=np.float64, copy=True, order="C", subok=False)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _probability_array(
    value: object,
    name: str = "p_pre_zero",
    *,
    ndim: int | None = None,
) -> np.ndarray:
    result = _numeric_array(value, name, ndim=ndim)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _binary_target(value: object) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise TypeError("target must not be a masked array")
    coerced = np.asanyarray(value)
    if np.ma.isMaskedArray(coerced):
        raise TypeError("target must not be a masked array")
    if coerced.dtype.metadata is not None:
        raise TypeError("target dtype metadata is not supported")
    if coerced.ndim != 1 or coerced.size == 0:
        raise ValueError("target must be a nonempty one-dimensional array")
    if coerced.dtype.kind not in "iu" or coerced.dtype.kind == "b":
        raise TypeError("target must contain integer binary exact-truth labels")
    if np.any((coerced != 0) & (coerced != 1)):
        raise ValueError("target must contain only 0 and 1")
    result = np.array(coerced, dtype=np.int8, copy=True, order="C", subok=False)
    return result


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """One exact-truth observed-zero development record."""

    p_pre_zero: tuple[float, ...]
    target: tuple[int, ...]
    mechanism: str
    biological_id: str
    manifest_sha256: str
    truth_kind: str

    def __post_init__(self) -> None:
        probability = _probability_array(self.p_pre_zero, ndim=1)
        target = _binary_target(self.target)
        if probability.shape != target.shape:
            raise ValueError("p_pre_zero and target lengths must match")
        if not isinstance(self.mechanism, str) or not _MECHANISM.fullmatch(
            self.mechanism
        ):
            raise ValueError("mechanism must be a canonical lowercase identifier")
        if self.mechanism != "symsim":
            raise ValueError(
                "only symsim supplies exact pre-capture truth in the current panel"
            )
        if not isinstance(self.biological_id, str) or not _BIOLOGICAL_ID.fullmatch(
            self.biological_id
        ):
            raise ValueError("biological_id must use canonical draw-{index:02d} form")
        if not isinstance(self.manifest_sha256, str) or not _SHA256.fullmatch(
            self.manifest_sha256
        ):
            raise ValueError("manifest_sha256 must be lowercase SHA-256")
        if self.truth_kind != "exact_pre_capture":
            raise ValueError("truth_kind must be exact_pre_capture")
        object.__setattr__(self, "p_pre_zero", tuple(float(x) for x in probability))
        object.__setattr__(self, "target", tuple(int(x) for x in target))


def _finite_coefficient(value: object, name: str, *, nonnegative: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class ScoreCalibrator:
    """Immutable monotone transformation of a count-derived score only."""

    algorithm: str
    coefficients: tuple[float, ...] = ()
    knots: tuple[float, ...] = ()
    values: tuple[float, ...] = ()

    _PARAMETER_COUNTS: ClassVar[dict[str, int]] = {
        "identity": 0,
        "logistic": 2,
        "beta": 3,
        "isotonic": 0,
    }

    def __post_init__(self) -> None:
        if self.algorithm not in self._PARAMETER_COUNTS:
            raise ValueError("unsupported calibration algorithm")
        coefficients = tuple(
            _finite_coefficient(value, "coefficient", nonnegative=False)
            for value in self.coefficients
        )
        knots = tuple(
            _finite_coefficient(value, "isotonic knot", nonnegative=False)
            for value in self.knots
        )
        values = tuple(
            _finite_coefficient(value, "isotonic value", nonnegative=False)
            for value in self.values
        )
        if self.algorithm in {"identity", "logistic", "beta"} and (knots or values):
            raise ValueError("parametric calibrators cannot contain isotonic knots")
        if len(coefficients) != self._PARAMETER_COUNTS[self.algorithm]:
            raise ValueError("wrong coefficient count for calibration algorithm")
        if self.algorithm == "logistic" and coefficients[1] < 0:
            raise ValueError("logistic slope must be nonnegative")
        if self.algorithm == "beta" and (coefficients[0] < 0 or coefficients[1] < 0):
            raise ValueError("beta a and b must be nonnegative")
        if self.algorithm == "isotonic":
            if coefficients or not knots or len(knots) != len(values):
                raise ValueError("isotonic calibrator requires paired knots and values")
            if not all(math.isfinite(value) for value in (*knots, *values)):
                raise ValueError("isotonic knots and values must be finite")
            if any(not 0 <= value <= 1 for value in (*knots, *values)):
                raise ValueError("isotonic knots and values must lie in [0, 1]")
            if any(right <= left for left, right in zip(knots, knots[1:])):
                raise ValueError("isotonic knots must be strictly increasing")
            if any(right < left for left, right in zip(values, values[1:])):
                raise ValueError("isotonic values must be nondecreasing")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "values", values)

    @classmethod
    def identity(cls) -> ScoreCalibrator:
        return cls("identity")

    @classmethod
    def logistic(cls, *, intercept: float, slope: float) -> ScoreCalibrator:
        return cls("logistic", (intercept, slope))

    @classmethod
    def beta(
        cls,
        *,
        a: float,
        b: float,
        intercept: float,
    ) -> ScoreCalibrator:
        return cls("beta", (a, b, intercept))

    @classmethod
    def isotonic(
        cls,
        *,
        knots: tuple[float, ...],
        values: tuple[float, ...],
    ) -> ScoreCalibrator:
        return cls("isotonic", (), knots, values)

    def transform(self, p_pre_zero: object) -> np.ndarray:
        probability = _probability_array(p_pre_zero)
        if self.algorithm == "identity":
            return probability
        clipped = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
        if self.algorithm == "logistic":
            intercept, slope = self.coefficients
            return expit(intercept + slope * logit(clipped))
        if self.algorithm == "beta":
            a, b, intercept = self.coefficients
            return expit(intercept + a * np.log(clipped) - b * np.log1p(-clipped))
        flattened = np.interp(
            probability.ravel(),
            np.asarray(self.knots),
            np.asarray(self.values),
        )
        return flattened.reshape(probability.shape)

    def to_dict(self) -> dict[str, Any]:
        if self.algorithm == "identity":
            parameters: dict[str, Any] = {}
        elif self.algorithm == "logistic":
            parameters = {
                "intercept": self.coefficients[0],
                "slope": self.coefficients[1],
            }
        elif self.algorithm == "beta":
            parameters = {
                "a": self.coefficients[0],
                "b": self.coefficients[1],
                "intercept": self.coefficients[2],
            }
        else:
            parameters = {"knots": list(self.knots), "values": list(self.values)}
        return {"algorithm": self.algorithm, "parameters": parameters}


def _validate_fit_arrays(
    p_pre_zero: object,
    target: object,
    weights: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = _probability_array(p_pre_zero, ndim=1)
    labels = _binary_target(target).astype(np.float64)
    sample_weights = _numeric_array(weights, "weights", ndim=1)
    if probability.shape != labels.shape or probability.shape != sample_weights.shape:
        raise ValueError("p_pre_zero, target, and weights lengths must match")
    if np.any(sample_weights <= 0):
        raise ValueError("weights must be strictly positive")
    return probability, labels, sample_weights


def _fit_parametric(
    algorithm: str,
    probability: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> ScoreCalibrator:
    if np.unique(target).size != 2:
        raise ValueError("parametric calibration requires both target classes")
    clipped = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    prevalence = float(np.sum(weights * target) / np.sum(weights))
    initial_intercept = float(
        logit(np.clip(prevalence, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
    )
    if algorithm == "logistic":
        design = np.column_stack((np.ones(len(clipped)), logit(clipped)))
        initial = np.array([initial_intercept, 1.0])
        bounds = ((None, None), (0.0, None))
    else:
        design = np.column_stack(
            (np.ones(len(clipped)), np.log(clipped), -np.log1p(-clipped))
        )
        initial = np.array([initial_intercept, 1.0, 1.0])
        bounds = ((None, None), (0.0, None), (0.0, None))
    total_weight = float(np.sum(weights))

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ coefficient
        loss = np.sum(weights * (np.logaddexp(0.0, linear) - target * linear))
        gradient = design.T @ (weights * (expit(linear) - target))
        return float(loss / total_weight), gradient / total_weight

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000, "maxls": 50},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"{algorithm} calibration optimization failed: {result.message}"
        )
    if algorithm == "logistic":
        return ScoreCalibrator.logistic(
            intercept=float(result.x[0]),
            slope=float(result.x[1]),
        )
    return ScoreCalibrator.beta(
        a=float(result.x[1]),
        b=float(result.x[2]),
        intercept=float(result.x[0]),
    )


def _fit_isotonic(
    probability: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> ScoreCalibrator:
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_target = target[order]
    sorted_weights = weights[order]
    knots, inverse = np.unique(sorted_probability, return_inverse=True)
    knot_weights = np.bincount(inverse, weights=sorted_weights)
    knot_positive = np.bincount(inverse, weights=sorted_weights * sorted_target)
    means = knot_positive / knot_weights

    blocks: list[list[float | int]] = []
    for index, (weight, mean) in enumerate(zip(knot_weights, means)):
        blocks.append([index, index, float(weight), float(mean)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left[2]) + float(right[2])
            merged_mean = (
                float(left[2]) * float(left[3]) + float(right[2]) * float(right[3])
            ) / merged_weight
            blocks.append([int(left[0]), int(right[1]), merged_weight, merged_mean])
    fitted = np.empty(len(knots), dtype=np.float64)
    for start, end, _, mean in blocks:
        fitted[int(start) : int(end) + 1] = float(mean)
    return ScoreCalibrator.isotonic(
        knots=tuple(float(value) for value in knots),
        values=tuple(float(value) for value in fitted),
    )


def fit_score_calibrator(
    algorithm: str,
    p_pre_zero: object,
    target: object,
    weights: object,
) -> ScoreCalibrator:
    """Fit one prespecified monotone calibrator by weighted log loss or PAV."""

    if algorithm not in CALIBRATOR_ORDER:
        raise ValueError("unsupported calibration algorithm")
    probability, labels, sample_weights = _validate_fit_arrays(
        p_pre_zero,
        target,
        weights,
    )
    if algorithm == "identity":
        return ScoreCalibrator.identity()
    if algorithm == "isotonic":
        return _fit_isotonic(probability, labels, sample_weights)
    return _fit_parametric(algorithm, probability, labels, sample_weights)


def validate_calibration_records(
    records: object,
) -> tuple[CalibrationRecord, ...]:
    """Return records in canonical order after rejecting duplicate manifests."""

    if isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of CalibrationRecord values")
    try:
        values = tuple(records)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "records must be an iterable of CalibrationRecord values"
        ) from exc
    if not values:
        raise ValueError("at least one calibration record is required")
    if any(not isinstance(record, CalibrationRecord) for record in values):
        raise TypeError("records must contain only CalibrationRecord values")
    manifests = [record.manifest_sha256 for record in values]
    if len(set(manifests)) != len(manifests):
        raise ValueError("duplicate calibration record manifest_sha256")
    return tuple(
        sorted(
            values,
            key=lambda record: (
                record.mechanism,
                record.biological_id,
                record.manifest_sha256,
            ),
        )
    )


def development_weights(
    records: object,
) -> dict[str, tuple[float, ...]]:
    """Assign equal mechanism, draw, record, and within-record influence."""

    canonical = validate_calibration_records(records)
    mechanisms = sorted({record.mechanism for record in canonical})
    result: dict[str, tuple[float, ...]] = {}
    for mechanism in mechanisms:
        mechanism_records = [
            record for record in canonical if record.mechanism == mechanism
        ]
        draws = sorted({record.biological_id for record in mechanism_records})
        for biological_id in draws:
            draw_records = [
                record
                for record in mechanism_records
                if record.biological_id == biological_id
            ]
            record_total = 1.0 / (len(mechanisms) * len(draws) * len(draw_records))
            for record in draw_records:
                entry_weight = record_total / len(record.p_pre_zero)
                result[record.manifest_sha256] = (entry_weight,) * len(
                    record.p_pre_zero
                )
    return result


def _stack_records(
    records: tuple[CalibrationRecord, ...],
    weights: dict[str, tuple[float, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.concatenate(
        [np.asarray(record.p_pre_zero, dtype=np.float64) for record in records]
    )
    target = np.concatenate(
        [np.asarray(record.target, dtype=np.int8) for record in records]
    )
    sample_weights = np.concatenate(
        [np.asarray(weights[record.manifest_sha256]) for record in records]
    )
    return probability, target, sample_weights


@dataclass(frozen=True, slots=True)
class CalibrationFold:
    mechanism: str
    biological_id: str
    held_out_manifests: tuple[str, ...]
    training_manifests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    algorithm: str
    predictions: tuple[tuple[str, tuple[float, ...]], ...]
    folds: tuple[CalibrationFold, ...]
    fit_failures: tuple[str, ...]


def cross_validate_calibrator(
    records: object,
    algorithm: str,
) -> CrossValidationResult:
    """Generate deterministic leave-one-mechanism-draw-out predictions."""

    if algorithm not in CALIBRATOR_ORDER:
        raise ValueError("unsupported calibration algorithm")
    canonical = validate_calibration_records(records)
    groups = sorted({(record.mechanism, record.biological_id) for record in canonical})
    if len(groups) < 2:
        raise ValueError("LODO calibration requires at least two biological draws")
    prediction_by_manifest: dict[str, tuple[float, ...]] = {}
    folds: list[CalibrationFold] = []
    failures: list[str] = []
    for mechanism, biological_id in groups:
        held_out = tuple(
            record
            for record in canonical
            if (record.mechanism, record.biological_id) == (mechanism, biological_id)
        )
        training = tuple(record for record in canonical if record not in held_out)
        training_weights = development_weights(training)
        train_probability, train_target, train_weight = _stack_records(
            training,
            training_weights,
        )
        try:
            calibrator = fit_score_calibrator(
                algorithm,
                train_probability,
                train_target,
                train_weight,
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            failures.append(f"{mechanism}/{biological_id}:{type(exc).__name__}:{exc}")
            calibrator = ScoreCalibrator.identity()
        for record in held_out:
            predicted = calibrator.transform(record.p_pre_zero)
            prediction_by_manifest[record.manifest_sha256] = tuple(
                float(value) for value in predicted
            )
        folds.append(
            CalibrationFold(
                mechanism=mechanism,
                biological_id=biological_id,
                held_out_manifests=tuple(record.manifest_sha256 for record in held_out),
                training_manifests=tuple(record.manifest_sha256 for record in training),
            )
        )
    return CrossValidationResult(
        algorithm=algorithm,
        predictions=tuple(sorted(prediction_by_manifest.items())),
        folds=tuple(folds),
        fit_failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    brier: float
    log_loss: float
    calibration_intercept: float | None
    calibration_slope: float | None
    slope_reason: str | None
    n: int

    def __post_init__(self) -> None:
        for name in ("brier", "log_loss"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.brier > 1:
            raise ValueError("brier must not exceed one")
        if type(self.n) is not int or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if (self.calibration_intercept is None) != (self.calibration_slope is None):
            raise ValueError("calibration intercept and slope availability differs")
        if self.calibration_intercept is None:
            if not isinstance(self.slope_reason, str) or not self.slope_reason:
                raise ValueError("undefined calibration slope requires a reason")
        else:
            if self.slope_reason is not None:
                raise ValueError("defined calibration slope cannot have a reason")
            for name in ("calibration_intercept", "calibration_slope"):
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"{name} must be finite")
                object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "brier": self.brier,
            "log_loss": self.log_loss,
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "slope_reason": self.slope_reason,
            "n": self.n,
        }


def _calibration_line(
    probability: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[float | None, float | None, str | None]:
    if np.unique(target).size != 2:
        return None, None, "single_target_class"
    predictor = logit(np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
    if np.ptp(predictor) == 0:
        return None, None, "constant_prediction"
    design = np.column_stack((np.ones(len(predictor)), predictor))
    total_weight = float(np.sum(weights))

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ coefficient
        loss = np.sum(weights * (np.logaddexp(0.0, linear) - target * linear))
        gradient = design.T @ (weights * (expit(linear) - target))
        return float(loss / total_weight), gradient / total_weight

    result = minimize(
        objective,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000, "maxls": 50},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None, None, "calibration_line_fit_failed"
    return float(result.x[0]), float(result.x[1]), None


def calibration_metrics(
    p_pre_zero: object,
    target: object,
    weights: object,
) -> CalibrationMetrics:
    """Compute prespecified weighted calibration metrics."""

    probability, labels, sample_weights = _validate_fit_arrays(
        p_pre_zero,
        target,
        weights,
    )
    normalized_weights = sample_weights / np.sum(sample_weights)
    brier = float(np.sum(normalized_weights * (probability - labels) ** 2))
    clipped = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    log_loss = float(
        -np.sum(
            normalized_weights
            * (labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        )
    )
    intercept, slope, reason = _calibration_line(
        probability,
        labels,
        sample_weights,
    )
    return CalibrationMetrics(
        brier=brier,
        log_loss=log_loss,
        calibration_intercept=intercept,
        calibration_slope=slope,
        slope_reason=reason,
        n=len(probability),
    )


@dataclass(frozen=True, slots=True)
class CalibrationThresholds:
    minimum_mechanisms_improved: int = 3
    brier_improvement_epsilon: float = 1e-6
    log_loss_worsening_tolerance: float = 1e-3
    calibration_slope_lower: float = 0.8
    calibration_slope_upper: float = 1.2

    def __post_init__(self) -> None:
        if (
            type(self.minimum_mechanisms_improved) is not int
            or self.minimum_mechanisms_improved <= 0
        ):
            raise ValueError("minimum_mechanisms_improved must be positive")
        for name in (
            "brier_improvement_epsilon",
            "log_loss_worsening_tolerance",
            "calibration_slope_lower",
            "calibration_slope_upper",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.calibration_slope_lower > self.calibration_slope_upper:
            raise ValueError("calibration slope bounds are reversed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_mechanisms_improved": self.minimum_mechanisms_improved,
            "brier_improvement_epsilon": self.brier_improvement_epsilon,
            "log_loss_worsening_tolerance": self.log_loss_worsening_tolerance,
            "calibration_slope_lower": self.calibration_slope_lower,
            "calibration_slope_upper": self.calibration_slope_upper,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    algorithm: str
    mechanism_metrics: tuple[tuple[str, CalibrationMetrics], ...]
    aggregate_metrics: CalibrationMetrics
    fit_failures: tuple[str, ...]
    brier_improved_mechanisms: tuple[str, ...]
    eligible: bool
    eligibility_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.algorithm not in CALIBRATOR_ORDER:
            raise ValueError("unsupported candidate algorithm")
        names = [name for name, _ in self.mechanism_metrics]
        if not names or names != sorted(names) or len(set(names)) != len(names):
            raise ValueError("mechanism_metrics must be unique and canonically sorted")
        if any(
            not isinstance(metric, CalibrationMetrics)
            for _, metric in self.mechanism_metrics
        ):
            raise TypeError("mechanism_metrics contains an invalid value")
        if not isinstance(self.aggregate_metrics, CalibrationMetrics):
            raise TypeError("aggregate_metrics must be CalibrationMetrics")
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be boolean")
        for values, name in (
            (self.fit_failures, "fit_failures"),
            (self.brier_improved_mechanisms, "brier_improved_mechanisms"),
            (self.eligibility_reasons, "eligibility_reasons"),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain nonempty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "mechanism_metrics": {
                name: metric.to_dict() for name, metric in self.mechanism_metrics
            },
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "fit_failures": list(self.fit_failures),
            "brier_improved_mechanisms": list(self.brier_improved_mechanisms),
            "eligible": self.eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
        }


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    selected_algorithm: str
    candidates: tuple[CandidateEvaluation, ...]

    def __post_init__(self) -> None:
        if self.selected_algorithm not in CALIBRATOR_ORDER:
            raise ValueError("selected_algorithm is invalid")
        if (
            len(self.candidates) != len(CALIBRATOR_ORDER)
            or tuple(candidate.algorithm for candidate in self.candidates)
            != CALIBRATOR_ORDER
        ):
            raise ValueError("decision must retain candidates in canonical order")
        selected = self.candidates[CALIBRATOR_ORDER.index(self.selected_algorithm)]
        if not selected.eligible:
            raise ValueError("selected calibration candidate is ineligible")


def _evaluate_cross_validation(
    records: tuple[CalibrationRecord, ...],
    result: CrossValidationResult,
) -> CandidateEvaluation:
    prediction_by_manifest = dict(result.predictions)
    weights = development_weights(records)
    mechanism_metrics: list[tuple[str, CalibrationMetrics]] = []
    all_probability: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for mechanism in sorted({record.mechanism for record in records}):
        mechanism_records = tuple(
            record for record in records if record.mechanism == mechanism
        )
        probability = np.concatenate(
            [
                np.asarray(prediction_by_manifest[record.manifest_sha256])
                for record in mechanism_records
            ]
        )
        target = np.concatenate(
            [np.asarray(record.target) for record in mechanism_records]
        )
        sample_weights = np.concatenate(
            [
                np.asarray(weights[record.manifest_sha256])
                for record in mechanism_records
            ]
        )
        mechanism_metrics.append(
            (mechanism, calibration_metrics(probability, target, sample_weights))
        )
        all_probability.append(probability)
        all_target.append(target)
        all_weights.append(sample_weights)
    aggregate = calibration_metrics(
        np.concatenate(all_probability),
        np.concatenate(all_target),
        np.concatenate(all_weights),
    )
    return CandidateEvaluation(
        algorithm=result.algorithm,
        mechanism_metrics=tuple(mechanism_metrics),
        aggregate_metrics=aggregate,
        fit_failures=result.fit_failures,
        brier_improved_mechanisms=(),
        eligible=result.algorithm == "identity",
        eligibility_reasons=("default_uncalibrated_score",)
        if result.algorithm == "identity"
        else (),
    )


def retention_reasons(
    candidate: CandidateEvaluation,
    identity: CandidateEvaluation,
    thresholds: CalibrationThresholds,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply the prespecified non-identity retention gate."""

    if candidate.algorithm == "identity":
        raise ValueError("identity is the default and does not pass a retention gate")
    candidate_metrics = dict(candidate.mechanism_metrics)
    identity_metrics = dict(identity.mechanism_metrics)
    if set(candidate_metrics) != set(identity_metrics):
        raise ValueError("candidate and identity mechanism sets differ")
    reasons = [f"fold_fit_failure:{failure}" for failure in candidate.fit_failures]
    if len(identity_metrics) < thresholds.minimum_mechanisms_improved:
        reasons.append(
            "insufficient_eligible_exact_truth_mechanisms:"
            f"{len(identity_metrics)}<{thresholds.minimum_mechanisms_improved}"
        )
    improved = tuple(
        mechanism
        for mechanism in sorted(identity_metrics)
        if candidate_metrics[mechanism].brier
        <= identity_metrics[mechanism].brier - thresholds.brier_improvement_epsilon
    )
    if len(improved) < thresholds.minimum_mechanisms_improved:
        reasons.append(
            "insufficient_mechanism_brier_improvement:"
            f"{len(improved)}<{thresholds.minimum_mechanisms_improved}"
        )
    if (
        candidate.aggregate_metrics.log_loss
        > identity.aggregate_metrics.log_loss + thresholds.log_loss_worsening_tolerance
    ):
        reasons.append("aggregate_log_loss_worsened")
    for mechanism in sorted(identity_metrics):
        metrics = candidate_metrics[mechanism]
        if (
            metrics.log_loss
            > identity_metrics[mechanism].log_loss
            + thresholds.log_loss_worsening_tolerance
        ):
            reasons.append(f"mechanism_log_loss_worsened:{mechanism}")
        if metrics.calibration_slope is None or not (
            thresholds.calibration_slope_lower
            <= metrics.calibration_slope
            <= thresholds.calibration_slope_upper
        ):
            reasons.append(f"mechanism_calibration_slope_outside_tolerance:{mechanism}")
    aggregate_slope = candidate.aggregate_metrics.calibration_slope
    if aggregate_slope is None or not (
        thresholds.calibration_slope_lower
        <= aggregate_slope
        <= thresholds.calibration_slope_upper
    ):
        reasons.append("aggregate_calibration_slope_outside_tolerance")
    return tuple(reasons), improved


def select_candidate(
    evaluations: object,
) -> CalibrationDecision:
    """Choose deterministically while retaining every candidate report."""

    values = tuple(evaluations)  # type: ignore[arg-type]
    if any(not isinstance(value, CandidateEvaluation) for value in values):
        raise TypeError("evaluations must contain CandidateEvaluation values")
    by_algorithm = {value.algorithm: value for value in values}
    if len(by_algorithm) != len(values) or set(by_algorithm) != set(CALIBRATOR_ORDER):
        raise ValueError("evaluations must contain each candidate exactly once")
    ordered = tuple(by_algorithm[name] for name in CALIBRATOR_ORDER)
    eligible = [
        candidate
        for candidate in ordered
        if candidate.algorithm != "identity" and candidate.eligible
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                candidate.aggregate_metrics.brier,
                candidate.aggregate_metrics.log_loss,
                CALIBRATOR_ORDER.index(candidate.algorithm),
            ),
        ).algorithm
    else:
        selected = "identity"
    return CalibrationDecision(selected_algorithm=selected, candidates=ordered)


def evaluate_calibration_candidates(
    records: object,
    thresholds: CalibrationThresholds = CalibrationThresholds(),
) -> CalibrationDecision:
    """Cross-validate all candidates and apply the retention decision."""

    canonical = validate_calibration_records(records)
    initial = tuple(
        _evaluate_cross_validation(
            canonical,
            cross_validate_calibrator(canonical, algorithm),
        )
        for algorithm in CALIBRATOR_ORDER
    )
    identity = initial[0]
    evaluated = [identity]
    for candidate in initial[1:]:
        reasons, improved = retention_reasons(candidate, identity, thresholds)
        evaluated.append(
            CandidateEvaluation(
                algorithm=candidate.algorithm,
                mechanism_metrics=candidate.mechanism_metrics,
                aggregate_metrics=candidate.aggregate_metrics,
                fit_failures=candidate.fit_failures,
                brier_improved_mechanisms=improved,
                eligible=not reasons,
                eligibility_reasons=reasons,
            )
        )
    return select_candidate(tuple(evaluated))


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != expected:
        raise ValueError(f"{name} has missing or extra fields")
    return value


def _finite_json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _calibrator_from_dict(value: object) -> ScoreCalibrator:
    payload = _exact_keys(value, {"algorithm", "parameters"}, "calibrator")
    algorithm = payload["algorithm"]
    if algorithm not in CALIBRATOR_ORDER:
        raise ValueError("calibrator algorithm is invalid")
    parameters = payload["parameters"]
    if algorithm == "identity":
        _exact_keys(parameters, set(), "calibrator.parameters")
        return ScoreCalibrator.identity()
    if algorithm == "logistic":
        values = _exact_keys(
            parameters,
            {"intercept", "slope"},
            "calibrator.parameters",
        )
        return ScoreCalibrator.logistic(
            intercept=_finite_json_number(values["intercept"], "intercept"),
            slope=_finite_json_number(values["slope"], "slope"),
        )
    if algorithm == "beta":
        values = _exact_keys(
            parameters,
            {"a", "b", "intercept"},
            "calibrator.parameters",
        )
        return ScoreCalibrator.beta(
            a=_finite_json_number(values["a"], "a"),
            b=_finite_json_number(values["b"], "b"),
            intercept=_finite_json_number(values["intercept"], "intercept"),
        )
    values = _exact_keys(
        parameters,
        {"knots", "values"},
        "calibrator.parameters",
    )
    if not isinstance(values["knots"], list) or not isinstance(values["values"], list):
        raise ValueError("isotonic knots and values must be arrays")
    return ScoreCalibrator.isotonic(
        knots=tuple(
            _finite_json_number(item, "isotonic knot") for item in values["knots"]
        ),
        values=tuple(
            _finite_json_number(item, "isotonic value") for item in values["values"]
        ),
    )


def _metrics_from_dict(value: object, name: str) -> CalibrationMetrics:
    payload = _exact_keys(
        value,
        {
            "brier",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
            "slope_reason",
            "n",
        },
        name,
    )
    n = payload["n"]
    if type(n) is not int or n <= 0:
        raise ValueError(f"{name}.n must be a positive integer")
    intercept_value = payload["calibration_intercept"]
    slope_value = payload["calibration_slope"]
    reason = payload["slope_reason"]
    if (intercept_value is None) != (slope_value is None):
        raise ValueError(f"{name} calibration line fields disagree")
    if intercept_value is None:
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{name} missing slope reason")
        intercept = None
        slope = None
    else:
        if reason is not None:
            raise ValueError(f"{name} slope reason must be null")
        intercept = _finite_json_number(intercept_value, f"{name}.intercept")
        slope = _finite_json_number(slope_value, f"{name}.slope")
    return CalibrationMetrics(
        brier=_finite_json_number(payload["brier"], f"{name}.brier"),
        log_loss=_finite_json_number(payload["log_loss"], f"{name}.log_loss"),
        calibration_intercept=intercept,
        calibration_slope=slope,
        slope_reason=reason,
        n=n,
    )


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be an array of nonempty strings")
    return tuple(value)


def _candidate_from_dict(value: object) -> CandidateEvaluation:
    payload = _exact_keys(
        value,
        {
            "algorithm",
            "mechanism_metrics",
            "aggregate_metrics",
            "fit_failures",
            "brier_improved_mechanisms",
            "eligible",
            "eligibility_reasons",
        },
        "candidate",
    )
    mechanism_payload = payload["mechanism_metrics"]
    if not isinstance(mechanism_payload, dict) or not mechanism_payload:
        raise ValueError("candidate mechanism_metrics must be a nonempty object")
    mechanism_metrics = tuple(
        (
            name,
            _metrics_from_dict(metric, f"candidate.mechanism_metrics.{name}"),
        )
        for name, metric in sorted(mechanism_payload.items())
    )
    return CandidateEvaluation(
        algorithm=payload["algorithm"],
        mechanism_metrics=mechanism_metrics,
        aggregate_metrics=_metrics_from_dict(
            payload["aggregate_metrics"],
            "candidate.aggregate_metrics",
        ),
        fit_failures=_string_list(payload["fit_failures"], "fit_failures"),
        brier_improved_mechanisms=_string_list(
            payload["brier_improved_mechanisms"],
            "brier_improved_mechanisms",
        ),
        eligible=payload["eligible"],
        eligibility_reasons=_string_list(
            payload["eligibility_reasons"],
            "eligibility_reasons",
        ),
    )


def _thresholds_from_dict(value: object) -> CalibrationThresholds:
    payload = _exact_keys(
        value,
        {
            "minimum_mechanisms_improved",
            "brier_improvement_epsilon",
            "log_loss_worsening_tolerance",
            "calibration_slope_lower",
            "calibration_slope_upper",
        },
        "selection.thresholds",
    )
    return CalibrationThresholds(**payload)


def _validate_artifact_payload(value: object) -> tuple[dict[str, Any], ScoreCalibrator]:
    payload = _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "estimand",
            "inference_features",
            "selected_algorithm",
            "calibrator",
            "selection",
            "cross_validation",
            "training",
            "truth_eligibility",
            "payload_sha256",
        },
        "artifact",
    )
    digest = payload["payload_sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("payload_sha256 is invalid")
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    if _canonical_digest(unsigned) != digest:
        raise ValueError("artifact payload digest mismatch")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported calibration artifact schema")
    if payload["artifact_type"] != "maskimpute_prezero_calibration":
        raise ValueError("artifact_type is invalid")
    if payload["estimand"] != "pre_capture_zero_given_observed_zero":
        raise ValueError("calibration estimand is invalid")
    if payload["inference_features"] != ["p_pre_zero"]:
        raise ValueError("calibrator inference features are invalid")

    truth_eligibility = _exact_keys(
        payload["truth_eligibility"],
        {
            "accepted_truth_kind",
            "eligible_mechanisms",
            "eligible_mechanism_count",
            "minimum_mechanisms_required",
            "panel_limitations",
        },
        "truth_eligibility",
    )
    if truth_eligibility["accepted_truth_kind"] != "exact_pre_capture":
        raise ValueError("accepted calibration truth kind is invalid")
    if truth_eligibility["eligible_mechanisms"] != ["symsim"]:
        raise ValueError("eligible calibration mechanisms are invalid")
    if (
        type(truth_eligibility["eligible_mechanism_count"]) is not int
        or truth_eligibility["eligible_mechanism_count"] != 1
    ):
        raise ValueError("eligible calibration mechanism count is invalid")
    if (
        type(truth_eligibility["minimum_mechanisms_required"]) is not int
        or truth_eligibility["minimum_mechanisms_required"] <= 0
    ):
        raise ValueError("minimum calibration mechanism count is invalid")
    if truth_eligibility["panel_limitations"] != {
        "semisynthetic": "proxy_truth_not_exact",
        "sergio": "undefined_for_continuous_truth",
        "sparsim": "undefined_for_continuous_truth",
    }:
        raise ValueError("calibration panel limitations are invalid")
    calibrator = _calibrator_from_dict(payload["calibrator"])
    if payload["selected_algorithm"] != calibrator.algorithm:
        raise ValueError("selected algorithm does not match calibrator")

    selection = _exact_keys(
        payload["selection"],
        {"candidate_order", "thresholds", "candidates", "decision_reason"},
        "selection",
    )
    if selection["candidate_order"] != list(CALIBRATOR_ORDER):
        raise ValueError("candidate_order is invalid")
    _thresholds_from_dict(selection["thresholds"])
    if (
        selection["thresholds"]["minimum_mechanisms_improved"]
        != truth_eligibility["minimum_mechanisms_required"]
    ):
        raise ValueError("truth eligibility and selection threshold disagree")
    if not isinstance(selection["candidates"], list):
        raise ValueError("selection.candidates must be an array")
    candidates = tuple(_candidate_from_dict(item) for item in selection["candidates"])
    decision = select_candidate(candidates)
    if decision.selected_algorithm != payload["selected_algorithm"]:
        raise ValueError("selected algorithm contradicts candidate reports")
    if (
        not isinstance(selection["decision_reason"], str)
        or not selection["decision_reason"]
    ):
        raise ValueError("selection.decision_reason must be nonempty")

    cross_validation = _exact_keys(
        payload["cross_validation"],
        {"scheme", "weighting", "groups"},
        "cross_validation",
    )
    if cross_validation["scheme"] != "leave_one_mechanism_biological_draw_out":
        raise ValueError("cross-validation scheme is invalid")
    if cross_validation["weighting"] != "mechanism_draw_record_entry_balanced":
        raise ValueError("cross-validation weighting is invalid")
    if (
        not isinstance(cross_validation["groups"], list)
        or not cross_validation["groups"]
    ):
        raise ValueError("cross-validation groups must be a nonempty array")
    group_manifests: list[str] = []
    previous_group: tuple[str, str] | None = None
    for index, item in enumerate(cross_validation["groups"]):
        group = _exact_keys(
            item,
            {"mechanism", "biological_id", "manifest_sha256s"},
            f"cross_validation.groups[{index}]",
        )
        mechanism = group["mechanism"]
        biological_id = group["biological_id"]
        if not isinstance(mechanism, str) or not _MECHANISM.fullmatch(mechanism):
            raise ValueError("cross-validation mechanism is invalid")
        if not isinstance(biological_id, str) or not _BIOLOGICAL_ID.fullmatch(
            biological_id
        ):
            raise ValueError("cross-validation biological_id is invalid")
        group_key = (mechanism, biological_id)
        if previous_group is not None and group_key <= previous_group:
            raise ValueError("cross-validation groups are not canonical")
        previous_group = group_key
        manifests = _string_list(
            group["manifest_sha256s"],
            "cross-validation manifest_sha256s",
        )
        if tuple(sorted(manifests)) != manifests or any(
            not _SHA256.fullmatch(manifest) for manifest in manifests
        ):
            raise ValueError("cross-validation manifest hashes are invalid")
        group_manifests.extend(manifests)

    training = _exact_keys(
        payload["training"],
        {
            "record_count",
            "entry_count",
            "record_digest_sha256",
            "manifest_sha256s",
        },
        "training",
    )
    if type(training["record_count"]) is not int or training["record_count"] <= 0:
        raise ValueError("training.record_count must be positive")
    if type(training["entry_count"]) is not int or training["entry_count"] <= 0:
        raise ValueError("training.entry_count must be positive")
    if not isinstance(training["record_digest_sha256"], str) or not _SHA256.fullmatch(
        training["record_digest_sha256"]
    ):
        raise ValueError("training record digest is invalid")
    manifests = _string_list(training["manifest_sha256s"], "training manifests")
    if tuple(sorted(manifests)) != manifests or len(set(manifests)) != len(manifests):
        raise ValueError("training manifests are not canonical and unique")
    if any(not _SHA256.fullmatch(manifest) for manifest in manifests):
        raise ValueError("training manifest hash is invalid")
    if training["record_count"] != len(manifests):
        raise ValueError("training record count does not match manifests")
    if tuple(sorted(group_manifests)) != manifests:
        raise ValueError("cross-validation and training manifests differ")
    return payload, calibrator


class CalibrationArtifact:
    """Canonical immutable fitted calibration artifact."""

    __slots__ = ("_payload_bytes", "_calibrator")

    def __init__(self, payload: object) -> None:
        validated, calibrator = _validate_artifact_payload(payload)
        object.__setattr__(self, "_payload_bytes", _canonical_json_bytes(validated))
        object.__setattr__(self, "_calibrator", calibrator)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot assign to immutable artifact field {name!r}")

    @property
    def selected_algorithm(self) -> str:
        return self._calibrator.algorithm

    def transform(self, p_pre_zero: object) -> np.ndarray:
        return self._calibrator.transform(p_pre_zero)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._payload_bytes)


def _record_payload(record: CalibrationRecord) -> dict[str, Any]:
    return {
        "p_pre_zero": list(record.p_pre_zero),
        "target": list(record.target),
        "mechanism": record.mechanism,
        "biological_id": record.biological_id,
        "manifest_sha256": record.manifest_sha256,
        "truth_kind": record.truth_kind,
    }


def fit_development_calibration(
    records: object,
    thresholds: CalibrationThresholds = CalibrationThresholds(),
) -> CalibrationArtifact:
    """Evaluate, select, and fit the retained calibrator on all development data."""

    canonical = validate_calibration_records(records)
    decision = evaluate_calibration_candidates(canonical, thresholds)
    weights = development_weights(canonical)
    probability, target, sample_weights = _stack_records(canonical, weights)
    calibrator = fit_score_calibrator(
        decision.selected_algorithm,
        probability,
        target,
        sample_weights,
    )
    groups = []
    for mechanism, biological_id in sorted(
        {(record.mechanism, record.biological_id) for record in canonical}
    ):
        groups.append(
            {
                "mechanism": mechanism,
                "biological_id": biological_id,
                "manifest_sha256s": sorted(
                    record.manifest_sha256
                    for record in canonical
                    if (record.mechanism, record.biological_id)
                    == (mechanism, biological_id)
                ),
            }
        )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "maskimpute_prezero_calibration",
        "estimand": "pre_capture_zero_given_observed_zero",
        "inference_features": ["p_pre_zero"],
        "truth_eligibility": {
            "accepted_truth_kind": "exact_pre_capture",
            "eligible_mechanisms": ["symsim"],
            "eligible_mechanism_count": 1,
            "minimum_mechanisms_required": thresholds.minimum_mechanisms_improved,
            "panel_limitations": {
                "semisynthetic": "proxy_truth_not_exact",
                "sergio": "undefined_for_continuous_truth",
                "sparsim": "undefined_for_continuous_truth",
            },
        },
        "selected_algorithm": decision.selected_algorithm,
        "calibrator": calibrator.to_dict(),
        "selection": {
            "candidate_order": list(CALIBRATOR_ORDER),
            "thresholds": thresholds.to_dict(),
            "candidates": [candidate.to_dict() for candidate in decision.candidates],
            "decision_reason": (
                "identity_default_no_nonidentity_passed"
                if decision.selected_algorithm == "identity"
                else "nonidentity_passed_prespecified_retention_gate"
            ),
        },
        "cross_validation": {
            "scheme": "leave_one_mechanism_biological_draw_out",
            "weighting": "mechanism_draw_record_entry_balanced",
            "groups": groups,
        },
        "training": {
            "record_count": len(canonical),
            "entry_count": sum(len(record.p_pre_zero) for record in canonical),
            "record_digest_sha256": _canonical_digest(
                [_record_payload(record) for record in canonical]
            ),
            "manifest_sha256s": sorted(record.manifest_sha256 for record in canonical),
        },
    }
    payload = dict(unsigned)
    payload["payload_sha256"] = _canonical_digest(unsigned)
    return CalibrationArtifact(payload)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_calibration_artifact(path: str | Path) -> CalibrationArtifact:
    """Load exact canonical bytes and verify the complete artifact schema/digest."""

    artifact_path = Path(path)
    try:
        raw = artifact_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid calibration artifact: {exc}") from exc
    artifact = CalibrationArtifact(payload)
    if raw != artifact._payload_bytes:
        raise ValueError("calibration artifact is not canonical JSON")
    return artifact


def load_calibration_records(path: str | Path) -> tuple[CalibrationRecord, ...]:
    """Load canonical exact-truth development calibration records."""

    input_path = Path(path)
    try:
        raw = input_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid calibration training input: {exc}") from exc
    if raw != _canonical_json_bytes(payload):
        raise ValueError("calibration training input is not canonical JSON")
    root = _exact_keys(
        payload,
        {"schema_version", "artifact_type", "records"},
        "training input",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("unsupported calibration training input schema")
    if root["artifact_type"] != "maskimpute_prezero_calibration_training_records":
        raise ValueError("calibration training artifact_type is invalid")
    if not isinstance(root["records"], list) or not root["records"]:
        raise ValueError("calibration training records must be a nonempty array")
    records: list[CalibrationRecord] = []
    for index, value in enumerate(root["records"]):
        record = _exact_keys(
            value,
            {
                "p_pre_zero",
                "target",
                "mechanism",
                "biological_id",
                "manifest_sha256",
                "truth_kind",
            },
            f"training input records[{index}]",
        )
        records.append(CalibrationRecord(**record))
    canonical = validate_calibration_records(records)
    if tuple(records) != canonical:
        raise ValueError("calibration training records are not in canonical order")
    return canonical


def save_calibration_artifact(
    path: str | Path,
    artifact: CalibrationArtifact,
) -> None:
    """Atomically publish canonical artifact bytes without overwriting."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(artifact._payload_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
