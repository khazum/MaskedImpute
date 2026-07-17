"""Leakage-safe development calibration for count-derived pre-zero scores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


PROBABILITY_CLIP = 1e-12
CALIBRATOR_ORDER = ("identity", "logistic", "beta", "isotonic")
DEVELOPMENT_PROTOCOL_SHA256 = (
    "7cfa1b55458b5b2bc4c22e3a155086724586d95df40aa61c4b78b1a779794249"
)
CALIBRATION_CONTRACT_SHA256 = (
    "180d85cc18e359970fff3c9cff37190c2b944b13b0883a46be2765c439a8a1b3"
)
_CALIBRATION_CONTRACT_ID = "prezero-calibration-retention-binding-v1"
_CALIBRATION_CONTRACT_PATH = "study/calibration_contract.json"
_DEVELOPMENT_NAMESPACE = "dev"
_DEVELOPMENT_DATA_ROLE = "development"
_DEVELOPMENT_BIOLOGICAL_IDS = ("draw-01", "draw-02")
_SYMSIM_TECHNICAL_VIEWS = ("moderate", "severe")
_DEVELOPMENT_PANEL_KEYS = frozenset(
    (biological_id, technical_view)
    for biological_id in _DEVELOPMENT_BIOLOGICAL_IDS
    for technical_view in _SYMSIM_TECHNICAL_VIEWS
)
_PRESPECIFIED_THRESHOLDS = {
    "minimum_exact_mechanisms_improved": 3,
    "minimum_biological_draws_improved": 2,
    "minimum_technical_records_improved": 4,
    "brier_improvement_epsilon": 1e-6,
    "log_loss_worsening_tolerance": 1e-3,
    "calibration_slope_lower": 0.8,
    "calibration_slope_upper": 1.2,
}
_MECHANISM = re.compile(r"[a-z][a-z0-9_]*")
_BIOLOGICAL_ID = re.compile(r"draw-(?:0[1-9]|[1-9][0-9])")
_DATASET_ID = re.compile(r"dataset-[0-9a-f]{24}")
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
    namespace: str
    data_role: str
    technical_view: str
    dataset_id: str
    dataset_sha256: str
    protocol_sha256: str

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
        if self.biological_id not in _DEVELOPMENT_BIOLOGICAL_IDS:
            raise ValueError(
                "biological_id is outside the prespecified development draw set"
            )
        if not isinstance(self.manifest_sha256, str) or not _SHA256.fullmatch(
            self.manifest_sha256
        ):
            raise ValueError("manifest_sha256 must be lowercase SHA-256")
        if self.truth_kind != "exact_pre_capture":
            raise ValueError("truth_kind must be exact_pre_capture")
        if self.namespace != _DEVELOPMENT_NAMESPACE:
            raise ValueError("calibration namespace must be the development namespace")
        if self.data_role != _DEVELOPMENT_DATA_ROLE:
            raise ValueError("calibration data_role must be development")
        if self.technical_view not in _SYMSIM_TECHNICAL_VIEWS:
            raise ValueError("technical_view is not a prespecified SymSim view")
        if not isinstance(self.dataset_id, str) or not _DATASET_ID.fullmatch(
            self.dataset_id
        ):
            raise ValueError("dataset_id must be a canonical simulation dataset ID")
        if not isinstance(self.dataset_sha256, str) or not _SHA256.fullmatch(
            self.dataset_sha256
        ):
            raise ValueError("dataset_sha256 must be lowercase SHA-256")
        if self.protocol_sha256 != DEVELOPMENT_PROTOCOL_SHA256:
            raise ValueError("protocol_sha256 does not bind the development protocol")
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


def _bounded_linear_predictor(
    features: np.ndarray,
    coefficients: tuple[float, ...],
) -> np.ndarray:
    """Evaluate a small linear model without overflowing finite coefficients."""

    coefficient = np.asarray(coefficients, dtype=np.float64)
    scale = float(np.max(np.abs(coefficient)))
    if scale == 0:
        return np.zeros(features.shape[0], dtype=np.float64)
    reduced = features @ (coefficient / scale)
    limit = np.finfo(np.float64).max / max(scale, 1.0)
    return np.clip(reduced, -limit, limit) * scale


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
            features = np.column_stack((np.ones(clipped.size), logit(clipped).ravel()))
            linear = _bounded_linear_predictor(features, (intercept, slope))
            return expit(linear).reshape(probability.shape)
        if self.algorithm == "beta":
            a, b, intercept = self.coefficients
            features = np.column_stack(
                (
                    np.log(clipped).ravel(),
                    -np.log1p(-clipped).ravel(),
                    np.ones(clipped.size),
                )
            )
            linear = _bounded_linear_predictor(features, (a, b, intercept))
            return expit(linear).reshape(probability.shape)
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
    scale = float(np.max(sample_weights))
    sample_weights /= scale
    total = float(np.sum(sample_weights))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("weights cannot be normalized safely")
    sample_weights /= total
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


def _canonical_calibration_records(
    records: object,
    *,
    require_complete_panel: bool,
) -> tuple[CalibrationRecord, ...]:
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
    if any(type(record) is not CalibrationRecord for record in values):
        raise TypeError("records must contain only CalibrationRecord values")
    manifests = [record.manifest_sha256 for record in values]
    if len(set(manifests)) != len(manifests):
        raise ValueError("duplicate calibration record manifest_sha256")
    dataset_ids = [record.dataset_id for record in values]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("duplicate calibration record dataset_id")
    dataset_sha256s = [record.dataset_sha256 for record in values]
    if len(set(dataset_sha256s)) != len(dataset_sha256s):
        raise ValueError("duplicate calibration record dataset_sha256")
    panel_keys = [(record.biological_id, record.technical_view) for record in values]
    if require_complete_panel and (
        len(panel_keys) != len(set(panel_keys))
        or set(panel_keys) != set(_DEVELOPMENT_PANEL_KEYS)
    ):
        raise ValueError(
            "calibration records must contain the complete prespecified draw-view panel"
        )
    return tuple(
        sorted(
            values,
            key=lambda record: (
                record.namespace,
                record.mechanism,
                record.biological_id,
                record.technical_view,
                record.dataset_id,
                record.manifest_sha256,
            ),
        )
    )


def validate_calibration_records(
    records: object,
) -> tuple[CalibrationRecord, ...]:
    """Return the complete development panel in canonical order."""

    return _canonical_calibration_records(records, require_complete_panel=True)


def _development_weights(
    records: object,
    *,
    require_complete_panel: bool,
) -> dict[str, tuple[float, ...]]:
    canonical = _canonical_calibration_records(
        records,
        require_complete_panel=require_complete_panel,
    )
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


def development_weights(
    records: object,
) -> dict[str, tuple[float, ...]]:
    """Assign equal mechanism, draw, record, and within-record influence."""

    return _development_weights(records, require_complete_panel=True)


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
    calibrator: ScoreCalibrator


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
        training_weights = _development_weights(
            training,
            require_complete_panel=False,
        )
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
                calibrator=calibrator,
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
    negative = predictor[target == 0]
    positive = predictor[target == 1]
    if np.max(negative) <= np.min(positive) or np.max(positive) <= np.min(negative):
        return None, None, "complete_or_quasi_separation"
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
    minimum_exact_mechanisms_improved: int = 3
    minimum_biological_draws_improved: int = 2
    minimum_technical_records_improved: int = 4
    brier_improvement_epsilon: float = 1e-6
    log_loss_worsening_tolerance: float = 1e-3
    calibration_slope_lower: float = 0.8
    calibration_slope_upper: float = 1.2

    def __post_init__(self) -> None:
        for name in (
            "minimum_exact_mechanisms_improved",
            "minimum_biological_draws_improved",
            "minimum_technical_records_improved",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
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
        if self.to_dict() != _PRESPECIFIED_THRESHOLDS:
            raise ValueError(
                "publication calibration thresholds must equal the prespecified values"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_exact_mechanisms_improved": (
                self.minimum_exact_mechanisms_improved
            ),
            "minimum_biological_draws_improved": (
                self.minimum_biological_draws_improved
            ),
            "minimum_technical_records_improved": (
                self.minimum_technical_records_improved
            ),
            "brier_improvement_epsilon": self.brier_improvement_epsilon,
            "log_loss_worsening_tolerance": self.log_loss_worsening_tolerance,
            "calibration_slope_lower": self.calibration_slope_lower,
            "calibration_slope_upper": self.calibration_slope_upper,
        }


def _validated_thresholds(value: object) -> CalibrationThresholds:
    if type(value) is not CalibrationThresholds:
        raise TypeError("thresholds must be an exact CalibrationThresholds value")
    return CalibrationThresholds(
        minimum_exact_mechanisms_improved=(
            value.minimum_exact_mechanisms_improved
        ),
        minimum_biological_draws_improved=value.minimum_biological_draws_improved,
        minimum_technical_records_improved=value.minimum_technical_records_improved,
        brier_improvement_epsilon=value.brier_improvement_epsilon,
        log_loss_worsening_tolerance=value.log_loss_worsening_tolerance,
        calibration_slope_lower=value.calibration_slope_lower,
        calibration_slope_upper=value.calibration_slope_upper,
    )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    algorithm: str
    mechanism_metrics: tuple[tuple[str, CalibrationMetrics], ...]
    biological_draw_metrics: tuple[tuple[str, CalibrationMetrics], ...]
    technical_record_metrics: tuple[tuple[str, CalibrationMetrics], ...]
    aggregate_metrics: CalibrationMetrics
    fit_failures: tuple[str, ...]
    brier_improved_mechanisms: tuple[str, ...]
    brier_improved_biological_draws: tuple[str, ...]
    brier_improved_technical_records: tuple[str, ...]
    eligible: bool
    eligibility_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.algorithm not in CALIBRATOR_ORDER:
            raise ValueError("unsupported candidate algorithm")
        metric_groups = (
            (self.mechanism_metrics, "mechanism_metrics"),
            (self.biological_draw_metrics, "biological_draw_metrics"),
            (self.technical_record_metrics, "technical_record_metrics"),
        )
        for metrics, name in metric_groups:
            names = [identifier for identifier, _metric in metrics]
            if (
                not names
                or any(not isinstance(identifier, str) or not identifier for identifier in names)
                or names != sorted(names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"{name} must be unique and canonically sorted")
            if any(not isinstance(metric, CalibrationMetrics) for _, metric in metrics):
                raise TypeError(f"{name} contains an invalid value")
        if not isinstance(self.aggregate_metrics, CalibrationMetrics):
            raise TypeError("aggregate_metrics must be CalibrationMetrics")
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be boolean")
        improvement_groups = (
            (
                self.brier_improved_mechanisms,
                {name for name, _metric in self.mechanism_metrics},
                "brier_improved_mechanisms",
            ),
            (
                self.brier_improved_biological_draws,
                {name for name, _metric in self.biological_draw_metrics},
                "brier_improved_biological_draws",
            ),
            (
                self.brier_improved_technical_records,
                {name for name, _metric in self.technical_record_metrics},
                "brier_improved_technical_records",
            ),
        )
        for identifiers, available, name in improvement_groups:
            if (
                tuple(sorted(identifiers)) != identifiers
                or len(set(identifiers)) != len(identifiers)
                or not set(identifiers).issubset(available)
            ):
                raise ValueError(f"{name} must be a canonical subset of metric IDs")
        for values, name in (
            (self.fit_failures, "fit_failures"),
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
            "biological_draw_metrics": {
                name: metric.to_dict()
                for name, metric in self.biological_draw_metrics
            },
            "technical_record_metrics": {
                name: metric.to_dict()
                for name, metric in self.technical_record_metrics
            },
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "fit_failures": list(self.fit_failures),
            "brier_improved_mechanisms": list(self.brier_improved_mechanisms),
            "brier_improved_biological_draws": list(
                self.brier_improved_biological_draws
            ),
            "brier_improved_technical_records": list(
                self.brier_improved_technical_records
            ),
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


def _technical_record_id(record: CalibrationRecord) -> str:
    return f"{record.mechanism}/{record.biological_id}/{record.technical_view}"


def _metrics_for_records(
    records: tuple[CalibrationRecord, ...],
    prediction_by_manifest: dict[str, tuple[float, ...]],
    weights: dict[str, tuple[float, ...]],
) -> CalibrationMetrics:
    probability = np.concatenate(
        [
            np.asarray(prediction_by_manifest[record.manifest_sha256])
            for record in records
        ]
    )
    target = np.concatenate([np.asarray(record.target) for record in records])
    sample_weights = np.concatenate(
        [np.asarray(weights[record.manifest_sha256]) for record in records]
    )
    return calibration_metrics(probability, target, sample_weights)


def _evaluate_cross_validation(
    records: tuple[CalibrationRecord, ...],
    result: CrossValidationResult,
) -> CandidateEvaluation:
    prediction_by_manifest = dict(result.predictions)
    weights = development_weights(records)
    mechanism_metrics = tuple(
        (
            mechanism,
            _metrics_for_records(
                tuple(record for record in records if record.mechanism == mechanism),
                prediction_by_manifest,
                weights,
            ),
        )
        for mechanism in sorted({record.mechanism for record in records})
    )
    biological_draw_metrics = tuple(
        (
            f"{mechanism}/{biological_id}",
            _metrics_for_records(
                tuple(
                    record
                    for record in records
                    if (record.mechanism, record.biological_id)
                    == (mechanism, biological_id)
                ),
                prediction_by_manifest,
                weights,
            ),
        )
        for mechanism, biological_id in sorted(
            {(record.mechanism, record.biological_id) for record in records}
        )
    )
    technical_record_metrics = tuple(
        (
            _technical_record_id(record),
            _metrics_for_records((record,), prediction_by_manifest, weights),
        )
        for record in records
    )
    return CandidateEvaluation(
        algorithm=result.algorithm,
        mechanism_metrics=mechanism_metrics,
        biological_draw_metrics=biological_draw_metrics,
        technical_record_metrics=technical_record_metrics,
        aggregate_metrics=_metrics_for_records(
            records,
            prediction_by_manifest,
            weights,
        ),
        fit_failures=result.fit_failures,
        brier_improved_mechanisms=(),
        brier_improved_biological_draws=(),
        brier_improved_technical_records=(),
        eligible=result.algorithm == "identity",
        eligibility_reasons=("default_uncalibrated_score",)
        if result.algorithm == "identity"
        else (),
    )


def retention_reasons(
    candidate: CandidateEvaluation,
    identity: CandidateEvaluation,
    thresholds: CalibrationThresholds,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Apply the prespecified non-identity retention gate."""

    thresholds = _validated_thresholds(thresholds)
    if candidate.algorithm == "identity":
        raise ValueError("identity is the default and does not pass a retention gate")
    candidate_metrics = dict(candidate.mechanism_metrics)
    identity_metrics = dict(identity.mechanism_metrics)
    if set(candidate_metrics) != set(identity_metrics):
        raise ValueError("candidate and identity mechanism sets differ")
    candidate_draw_metrics = dict(candidate.biological_draw_metrics)
    identity_draw_metrics = dict(identity.biological_draw_metrics)
    if set(candidate_draw_metrics) != set(identity_draw_metrics):
        raise ValueError("candidate and identity biological draw sets differ")
    candidate_record_metrics = dict(candidate.technical_record_metrics)
    identity_record_metrics = dict(identity.technical_record_metrics)
    if set(candidate_record_metrics) != set(identity_record_metrics):
        raise ValueError("candidate and identity technical record sets differ")
    reasons = [f"fold_fit_failure:{failure}" for failure in candidate.fit_failures]
    if len(identity_metrics) < thresholds.minimum_exact_mechanisms_improved:
        reasons.append(
            "insufficient_eligible_exact_truth_mechanisms:"
            f"{len(identity_metrics)}<"
            f"{thresholds.minimum_exact_mechanisms_improved}"
        )
    improved_mechanisms = tuple(
        mechanism
        for mechanism in sorted(identity_metrics)
        if candidate_metrics[mechanism].brier
        <= identity_metrics[mechanism].brier - thresholds.brier_improvement_epsilon
    )
    if len(improved_mechanisms) < thresholds.minimum_exact_mechanisms_improved:
        reasons.append(
            "insufficient_exact_mechanism_brier_improvement:"
            f"{len(improved_mechanisms)}<"
            f"{thresholds.minimum_exact_mechanisms_improved}"
        )
    missing_mechanisms = tuple(
        sorted(set(identity_metrics).difference(improved_mechanisms))
    )
    if missing_mechanisms:
        reasons.append(
            "not_all_exact_mechanisms_improved:" + ",".join(missing_mechanisms)
        )
    if len(identity_draw_metrics) < thresholds.minimum_biological_draws_improved:
        reasons.append(
            "insufficient_independent_biological_draws:"
            f"{len(identity_draw_metrics)}<"
            f"{thresholds.minimum_biological_draws_improved}"
        )
    improved_draws = tuple(
        identifier
        for identifier in sorted(identity_draw_metrics)
        if candidate_draw_metrics[identifier].brier
        <= identity_draw_metrics[identifier].brier
        - thresholds.brier_improvement_epsilon
    )
    if len(improved_draws) < thresholds.minimum_biological_draws_improved:
        reasons.append(
            "insufficient_biological_draw_brier_improvement:"
            f"{len(improved_draws)}<"
            f"{thresholds.minimum_biological_draws_improved}"
        )
    missing_draws = tuple(
        sorted(set(identity_draw_metrics).difference(improved_draws))
    )
    if missing_draws:
        reasons.append(
            "not_all_biological_draws_improved:" + ",".join(missing_draws)
        )
    if len(identity_record_metrics) < thresholds.minimum_technical_records_improved:
        reasons.append(
            "insufficient_nested_technical_records:"
            f"{len(identity_record_metrics)}<"
            f"{thresholds.minimum_technical_records_improved}"
        )
    improved_records = tuple(
        identifier
        for identifier in sorted(identity_record_metrics)
        if candidate_record_metrics[identifier].brier
        <= identity_record_metrics[identifier].brier
        - thresholds.brier_improvement_epsilon
    )
    if len(improved_records) < thresholds.minimum_technical_records_improved:
        reasons.append(
            "insufficient_technical_record_brier_improvement:"
            f"{len(improved_records)}<"
            f"{thresholds.minimum_technical_records_improved}"
        )
    missing_records = tuple(
        sorted(set(identity_record_metrics).difference(improved_records))
    )
    if missing_records:
        reasons.append(
            "not_all_technical_records_improved:" + ",".join(missing_records)
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
    for identifier in sorted(identity_draw_metrics):
        metrics = candidate_draw_metrics[identifier]
        if (
            metrics.log_loss
            > identity_draw_metrics[identifier].log_loss
            + thresholds.log_loss_worsening_tolerance
        ):
            reasons.append(f"biological_draw_log_loss_worsened:{identifier}")
        if metrics.calibration_slope is None or not (
            thresholds.calibration_slope_lower
            <= metrics.calibration_slope
            <= thresholds.calibration_slope_upper
        ):
            reasons.append(
                f"biological_draw_calibration_slope_outside_tolerance:{identifier}"
            )
    for identifier in sorted(identity_record_metrics):
        if (
            candidate_record_metrics[identifier].log_loss
            > identity_record_metrics[identifier].log_loss
            + thresholds.log_loss_worsening_tolerance
        ):
            reasons.append(f"technical_record_log_loss_worsened:{identifier}")
    aggregate_slope = candidate.aggregate_metrics.calibration_slope
    if aggregate_slope is None or not (
        thresholds.calibration_slope_lower
        <= aggregate_slope
        <= thresholds.calibration_slope_upper
    ):
        reasons.append("aggregate_calibration_slope_outside_tolerance")
    return (
        tuple(reasons),
        improved_mechanisms,
        improved_draws,
        improved_records,
    )


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

    thresholds = _validated_thresholds(thresholds)
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
        (
            reasons,
            improved_mechanisms,
            improved_draws,
            improved_records,
        ) = retention_reasons(candidate, identity, thresholds)
        evaluated.append(
            CandidateEvaluation(
                algorithm=candidate.algorithm,
                mechanism_metrics=candidate.mechanism_metrics,
                biological_draw_metrics=candidate.biological_draw_metrics,
                technical_record_metrics=candidate.technical_record_metrics,
                aggregate_metrics=candidate.aggregate_metrics,
                fit_failures=candidate.fit_failures,
                brier_improved_mechanisms=improved_mechanisms,
                brier_improved_biological_draws=improved_draws,
                brier_improved_technical_records=improved_records,
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
            "biological_draw_metrics",
            "technical_record_metrics",
            "aggregate_metrics",
            "fit_failures",
            "brier_improved_mechanisms",
            "brier_improved_biological_draws",
            "brier_improved_technical_records",
            "eligible",
            "eligibility_reasons",
        },
        "candidate",
    )
    def metric_group(field: str) -> tuple[tuple[str, CalibrationMetrics], ...]:
        values = payload[field]
        if not isinstance(values, dict) or not values:
            raise ValueError(f"candidate {field} must be a nonempty object")
        return tuple(
            (
                name,
                _metrics_from_dict(metric, f"candidate.{field}.{name}"),
            )
            for name, metric in sorted(values.items())
        )

    return CandidateEvaluation(
        algorithm=payload["algorithm"],
        mechanism_metrics=metric_group("mechanism_metrics"),
        biological_draw_metrics=metric_group("biological_draw_metrics"),
        technical_record_metrics=metric_group("technical_record_metrics"),
        aggregate_metrics=_metrics_from_dict(
            payload["aggregate_metrics"],
            "candidate.aggregate_metrics",
        ),
        fit_failures=_string_list(payload["fit_failures"], "fit_failures"),
        brier_improved_mechanisms=_string_list(
            payload["brier_improved_mechanisms"],
            "brier_improved_mechanisms",
        ),
        brier_improved_biological_draws=_string_list(
            payload["brier_improved_biological_draws"],
            "brier_improved_biological_draws",
        ),
        brier_improved_technical_records=_string_list(
            payload["brier_improved_technical_records"],
            "brier_improved_technical_records",
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
            "minimum_exact_mechanisms_improved",
            "minimum_biological_draws_improved",
            "minimum_technical_records_improved",
            "brier_improvement_epsilon",
            "log_loss_worsening_tolerance",
            "calibration_slope_lower",
            "calibration_slope_upper",
        },
        "selection.thresholds",
    )
    return CalibrationThresholds(**payload)


def _validate_artifact_payload(
    value: object,
) -> tuple[
    dict[str, Any],
    ScoreCalibrator,
    tuple[tuple[tuple[str, str], ScoreCalibrator], ...],
]:
    if (
        isinstance(value, dict)
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 2
    ):
        raise ValueError(
            "calibration artifact schema 2 is obsolete after the development "
            "retention contract"
        )
    payload = _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "estimand",
            "inference_features",
            "data_scope",
            "retention_contract",
            "selected_algorithm",
            "calibrator",
            "selection",
            "cross_validation",
            "development_holdout_calibrators",
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
    schema_version = payload["schema_version"]
    if type(schema_version) is not int:
        raise ValueError("unsupported calibration artifact schema")
    if schema_version != 3:
        raise ValueError("unsupported calibration artifact schema")
    if payload["artifact_type"] != "maskimpute_prezero_calibration":
        raise ValueError("artifact_type is invalid")
    if payload["estimand"] != "pre_capture_zero_given_observed_zero":
        raise ValueError("calibration estimand is invalid")
    if payload["inference_features"] != ["p_pre_zero"]:
        raise ValueError("calibrator inference features are invalid")
    data_scope = _exact_keys(
        payload["data_scope"],
        {
            "allowed_biological_ids",
            "data_role",
            "namespace",
            "protocol_sha256",
        },
        "data_scope",
    )
    if data_scope != {
        "allowed_biological_ids": list(_DEVELOPMENT_BIOLOGICAL_IDS),
        "data_role": _DEVELOPMENT_DATA_ROLE,
        "namespace": _DEVELOPMENT_NAMESPACE,
        "protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
    }:
        raise ValueError("calibration artifact is outside the development data scope")
    retention_contract = _exact_keys(
        payload["retention_contract"],
        {"contract_id", "path", "sha256"},
        "retention_contract",
    )
    if retention_contract != {
        "contract_id": _CALIBRATION_CONTRACT_ID,
        "path": _CALIBRATION_CONTRACT_PATH,
        "sha256": CALIBRATION_CONTRACT_SHA256,
    }:
        raise ValueError("calibration retention contract binding is invalid")

    truth_eligibility = _exact_keys(
        payload["truth_eligibility"],
        {
            "accepted_truth_kind",
            "eligible_mechanisms",
            "eligible_mechanism_count",
            "minimum_exact_mechanisms_improved",
            "minimum_biological_draws_improved",
            "minimum_technical_records_improved",
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
    for field in (
        "minimum_exact_mechanisms_improved",
        "minimum_biological_draws_improved",
        "minimum_technical_records_improved",
    ):
        if (
            type(truth_eligibility[field]) is not int
            or truth_eligibility[field] != _PRESPECIFIED_THRESHOLDS[field]
        ):
            raise ValueError(f"{field} is invalid")
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
    thresholds = _thresholds_from_dict(selection["thresholds"])
    for field in (
        "minimum_exact_mechanisms_improved",
        "minimum_biological_draws_improved",
        "minimum_technical_records_improved",
    ):
        if selection["thresholds"][field] != truth_eligibility[field]:
            raise ValueError("truth eligibility and selection threshold disagree")
    if not isinstance(selection["candidates"], list):
        raise ValueError("selection.candidates must be an array")
    candidates = tuple(_candidate_from_dict(item) for item in selection["candidates"])
    if tuple(candidate.algorithm for candidate in candidates) != CALIBRATOR_ORDER:
        raise ValueError("candidate reports are not in canonical algorithm order")
    expected_mechanisms = tuple(truth_eligibility["eligible_mechanisms"])
    if any(
        tuple(name for name, _metric in candidate.mechanism_metrics)
        != expected_mechanisms
        for candidate in candidates
    ):
        raise ValueError("candidate mechanisms contradict exact-truth eligibility")
    expected_draw_ids = tuple(
        f"symsim/{biological_id}"
        for biological_id in _DEVELOPMENT_BIOLOGICAL_IDS
    )
    expected_record_ids = tuple(
        sorted(
            f"symsim/{biological_id}/{technical_view}"
            for biological_id, technical_view in _DEVELOPMENT_PANEL_KEYS
        )
    )
    if any(
        tuple(name for name, _metric in candidate.biological_draw_metrics)
        != expected_draw_ids
        or tuple(name for name, _metric in candidate.technical_record_metrics)
        != expected_record_ids
        for candidate in candidates
    ):
        raise ValueError("candidate draw or technical record metrics are incomplete")
    for candidate in candidates:
        mechanism_entry_count = sum(
            metric.n for _name, metric in candidate.mechanism_metrics
        )
        if candidate.aggregate_metrics.n != mechanism_entry_count:
            raise ValueError(
                "candidate aggregate metric entry count contradicts mechanism metrics"
            )
        draw_entry_count = sum(
            metric.n for _name, metric in candidate.biological_draw_metrics
        )
        record_entry_count = sum(
            metric.n for _name, metric in candidate.technical_record_metrics
        )
        if candidate.aggregate_metrics.n not in {
            draw_entry_count,
            record_entry_count,
        } or draw_entry_count != record_entry_count:
            raise ValueError(
                "candidate aggregate count contradicts draw or technical record metrics"
            )
        record_metrics = dict(candidate.technical_record_metrics)
        for draw_id, draw_metric in candidate.biological_draw_metrics:
            nested_metrics = tuple(
                metric
                for record_id, metric in record_metrics.items()
                if record_id.startswith(f"{draw_id}/")
            )
            if draw_metric.n != sum(metric.n for metric in nested_metrics):
                raise ValueError(
                    "candidate biological draw count contradicts technical records"
                )
            for field in ("brier", "log_loss"):
                expected = math.fsum(
                    getattr(metric, field) for metric in nested_metrics
                ) / len(nested_metrics)
                if not math.isclose(
                    getattr(draw_metric, field),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "candidate biological draw metric contradicts equally "
                        "weighted technical records"
                    )
        mechanism_metric = candidate.mechanism_metrics[0][1]
        for field in ("brier", "log_loss"):
            expected = math.fsum(
                getattr(metric, field) for metric in record_metrics.values()
            ) / len(record_metrics)
            if not math.isclose(
                getattr(mechanism_metric, field),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(
                getattr(candidate.aggregate_metrics, field),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "candidate aggregate or mechanism metric contradicts equally "
                    "weighted technical records"
                )
    identity = candidates[0]
    if (
        not identity.eligible
        or identity.fit_failures
        or identity.brier_improved_mechanisms
        or identity.brier_improved_biological_draws
        or identity.brier_improved_technical_records
        or identity.eligibility_reasons != ("default_uncalibrated_score",)
    ):
        raise ValueError("identity candidate semantics are invalid")
    for candidate in candidates[1:]:
        (
            expected_reasons,
            expected_improved_mechanisms,
            expected_improved_draws,
            expected_improved_records,
        ) = retention_reasons(
            candidate,
            identity,
            thresholds,
        )
        if (
            candidate.eligible is not (not expected_reasons)
            or candidate.eligibility_reasons != expected_reasons
            or candidate.brier_improved_mechanisms
            != expected_improved_mechanisms
            or candidate.brier_improved_biological_draws != expected_improved_draws
            or candidate.brier_improved_technical_records != expected_improved_records
        ):
            raise ValueError(
                "candidate retention eligibility was not derived correctly"
            )
    decision = select_candidate(candidates)
    if decision.selected_algorithm != payload["selected_algorithm"]:
        raise ValueError("selected algorithm contradicts candidate reports")
    expected_decision_reason = (
        "identity_default_no_nonidentity_passed"
        if decision.selected_algorithm == "identity"
        else "nonidentity_passed_prespecified_retention_gate"
    )
    if selection["decision_reason"] != expected_decision_reason:
        raise ValueError("selection.decision_reason contradicts the retention decision")

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
    group_binding_by_manifest: dict[str, tuple[str, str, str]] = {}
    group_manifests_by_key: dict[tuple[str, str], tuple[str, ...]] = {}
    previous_group: tuple[str, str] | None = None
    for index, item in enumerate(cross_validation["groups"]):
        group = _exact_keys(
            item,
            {"namespace", "mechanism", "biological_id", "manifest_sha256s"},
            f"cross_validation.groups[{index}]",
        )
        namespace = group["namespace"]
        mechanism = group["mechanism"]
        biological_id = group["biological_id"]
        if namespace != _DEVELOPMENT_NAMESPACE:
            raise ValueError("cross-validation namespace is outside development")
        if mechanism not in expected_mechanisms:
            raise ValueError(
                "cross-validation mechanism is outside exact-truth eligibility"
            )
        if (
            not isinstance(biological_id, str)
            or not _BIOLOGICAL_ID.fullmatch(biological_id)
            or biological_id not in _DEVELOPMENT_BIOLOGICAL_IDS
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
        if any(manifest in group_binding_by_manifest for manifest in manifests):
            raise ValueError("cross-validation manifest is assigned to multiple groups")
        for manifest in manifests:
            group_binding_by_manifest[manifest] = (
                namespace,
                mechanism,
                biological_id,
            )
        group_manifests_by_key[group_key] = manifests
        group_manifests.extend(manifests)
    expected_group_keys = {
        ("symsim", biological_id) for biological_id in _DEVELOPMENT_BIOLOGICAL_IDS
    }
    if set(group_binding_by_manifest.values()) != {
        (_DEVELOPMENT_NAMESPACE, mechanism, biological_id)
        for mechanism, biological_id in expected_group_keys
    } or len(cross_validation["groups"]) != len(expected_group_keys):
        raise ValueError(
            "cross-validation groups do not cover the complete development draw panel"
        )

    training = _exact_keys(
        payload["training"],
        {
            "record_count",
            "entry_count",
            "record_digest_sha256",
            "record_bindings",
            "manifest_sha256s",
        },
        "training",
    )
    if type(training["record_count"]) is not int or training["record_count"] != len(
        _DEVELOPMENT_PANEL_KEYS
    ):
        raise ValueError(
            "training.record_count must match the complete draw-view panel"
        )
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
    if any(
        candidate.aggregate_metrics.n != training["entry_count"]
        for candidate in candidates
    ):
        raise ValueError(
            "training entry count does not match candidate aggregate metric counts"
        )
    if tuple(sorted(group_manifests)) != manifests:
        raise ValueError("cross-validation and training manifests differ")
    binding_payload = training["record_bindings"]
    if not isinstance(binding_payload, list) or len(binding_payload) != len(manifests):
        raise ValueError("training record bindings do not match record count")
    previous_binding_key: tuple[str, ...] | None = None
    binding_manifests: list[str] = []
    dataset_ids: list[str] = []
    dataset_sha256s: list[str] = []
    binding_panel_keys: list[tuple[str, str]] = []
    for index, value in enumerate(binding_payload):
        binding = _exact_keys(
            value,
            {
                "biological_id",
                "data_role",
                "dataset_id",
                "dataset_sha256",
                "manifest_sha256",
                "mechanism",
                "namespace",
                "protocol_sha256",
                "technical_view",
            },
            f"training.record_bindings[{index}]",
        )
        if (
            binding["namespace"] != _DEVELOPMENT_NAMESPACE
            or binding["data_role"] != _DEVELOPMENT_DATA_ROLE
            or binding["protocol_sha256"] != DEVELOPMENT_PROTOCOL_SHA256
            or binding["mechanism"] not in expected_mechanisms
            or binding["biological_id"] not in _DEVELOPMENT_BIOLOGICAL_IDS
            or binding["technical_view"] not in _SYMSIM_TECHNICAL_VIEWS
        ):
            raise ValueError("training record binding is outside development scope")
        dataset_id = binding["dataset_id"]
        dataset_sha256 = binding["dataset_sha256"]
        manifest_sha256 = binding["manifest_sha256"]
        if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
            raise ValueError("training record dataset_id is invalid")
        if not isinstance(dataset_sha256, str) or not _SHA256.fullmatch(dataset_sha256):
            raise ValueError("training record dataset_sha256 is invalid")
        if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(
            manifest_sha256
        ):
            raise ValueError("training record manifest_sha256 is invalid")
        binding_key = (
            binding["namespace"],
            binding["mechanism"],
            binding["biological_id"],
            binding["technical_view"],
            dataset_id,
            manifest_sha256,
        )
        if previous_binding_key is not None and binding_key <= previous_binding_key:
            raise ValueError("training record bindings are not canonical")
        previous_binding_key = binding_key
        if group_binding_by_manifest.get(manifest_sha256) != (
            binding["namespace"],
            binding["mechanism"],
            binding["biological_id"],
        ):
            raise ValueError("training binding contradicts cross-validation group")
        binding_manifests.append(manifest_sha256)
        dataset_ids.append(dataset_id)
        dataset_sha256s.append(dataset_sha256)
        binding_panel_keys.append((binding["biological_id"], binding["technical_view"]))
    if tuple(sorted(binding_manifests)) != manifests:
        raise ValueError("training record bindings differ from manifests")
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("training record bindings contain duplicate dataset IDs")
    if len(set(dataset_sha256s)) != len(dataset_sha256s):
        raise ValueError(
            "training record bindings contain duplicate dataset_sha256 values"
        )
    if len(binding_panel_keys) != len(set(binding_panel_keys)) or set(
        binding_panel_keys
    ) != set(_DEVELOPMENT_PANEL_KEYS):
        raise ValueError(
            "training record bindings do not cover the complete draw-view panel"
        )
    holdout_payload = payload["development_holdout_calibrators"]
    if not isinstance(holdout_payload, list):
        raise ValueError("development holdout calibrators must be an array")
    holdout_calibrators: list[tuple[tuple[str, str], ScoreCalibrator]] = []
    previous_holdout_key: tuple[str, str] | None = None
    for index, value in enumerate(holdout_payload):
        fold = _exact_keys(
            value,
            {
                "mechanism",
                "biological_id",
                "held_out_manifest_sha256s",
                "training_manifest_sha256s",
                "calibrator",
            },
            f"development_holdout_calibrators[{index}]",
        )
        key = (fold["mechanism"], fold["biological_id"])
        if key not in expected_group_keys or (
            previous_holdout_key is not None and key <= previous_holdout_key
        ):
            raise ValueError("development holdout calibrators are not canonical")
        previous_holdout_key = key
        held_out = _string_list(
            fold["held_out_manifest_sha256s"],
            "held_out_manifest_sha256s",
        )
        training_manifests = _string_list(
            fold["training_manifest_sha256s"],
            "training_manifest_sha256s",
        )
        if (
            tuple(sorted(held_out)) != held_out
            or tuple(sorted(training_manifests)) != training_manifests
            or len(set(held_out)) != len(held_out)
            or len(set(training_manifests)) != len(training_manifests)
            or any(not _SHA256.fullmatch(item) for item in (*held_out, *training_manifests))
        ):
            raise ValueError("development holdout fold manifests are not canonical")
        if held_out != group_manifests_by_key[key]:
            raise ValueError("development holdout manifests contradict fold group")
        held_set = set(held_out)
        training_set = set(training_manifests)
        if (
            not held_set.isdisjoint(training_set)
            or held_set | training_set != set(manifests)
            or training_set != set(manifests).difference(held_set)
        ):
            raise ValueError(
                "development holdout fold training must exclude held-out truth"
            )
        fold_calibrator = _calibrator_from_dict(fold["calibrator"])
        if fold_calibrator.algorithm != payload["selected_algorithm"]:
            raise ValueError(
                "development holdout calibrator algorithm differs from selection"
            )
        holdout_calibrators.append((key, fold_calibrator))
    if tuple(key for key, _calibrator in holdout_calibrators) != tuple(
        sorted(expected_group_keys)
    ):
        raise ValueError("development holdout calibrators do not cover every draw")
    return payload, calibrator, tuple(holdout_calibrators)


class CalibrationArtifact:
    """Canonical immutable fitted calibration artifact."""

    __slots__ = ("_payload_bytes", "_calibrator", "_holdout_calibrators")

    def __init__(self, payload: object) -> None:
        validated, calibrator, holdout_calibrators = _validate_artifact_payload(payload)
        object.__setattr__(self, "_payload_bytes", _canonical_json_bytes(validated))
        object.__setattr__(self, "_calibrator", calibrator)
        object.__setattr__(self, "_holdout_calibrators", holdout_calibrators)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot assign to immutable artifact field {name!r}")

    @property
    def selected_algorithm(self) -> str:
        return self._calibrator.algorithm

    def _verified_calibrators(
        self,
    ) -> tuple[
        ScoreCalibrator,
        tuple[tuple[tuple[str, str], ScoreCalibrator], ...],
    ]:
        try:
            payload = json.loads(self._payload_bytes)
            _validated, calibrator, holdout_calibrators = _validate_artifact_payload(
                payload
            )
        except Exception as exc:
            raise ValueError("calibration artifact integrity check failed") from exc
        if (
            calibrator != self._calibrator
            or holdout_calibrators != self._holdout_calibrators
        ):
            raise ValueError("calibration artifact integrity check failed")
        return calibrator, holdout_calibrators

    def transform(self, p_pre_zero: object) -> np.ndarray:
        calibrator, _holdout_calibrators = self._verified_calibrators()
        return calibrator.transform(p_pre_zero)

    def transform_for_development_holdout(
        self,
        p_pre_zero: object,
        *,
        mechanism: str,
        biological_id: str,
    ) -> np.ndarray:
        """Apply only the LODO calibrator that excluded this development draw."""

        if not isinstance(mechanism, str) or mechanism not in {"symsim"}:
            raise ValueError("mechanism is outside the development holdout scope")
        if (
            not isinstance(biological_id, str)
            or biological_id not in _DEVELOPMENT_BIOLOGICAL_IDS
        ):
            raise ValueError("biological_id is outside the development holdout scope")
        _calibrator, holdout_calibrators = self._verified_calibrators()
        by_group = dict(holdout_calibrators)
        try:
            fold_calibrator = by_group[(mechanism, biological_id)]
        except KeyError as exc:
            raise ValueError("development holdout calibrator is unavailable") from exc
        return fold_calibrator.transform(p_pre_zero)

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
        "namespace": record.namespace,
        "data_role": record.data_role,
        "technical_view": record.technical_view,
        "dataset_id": record.dataset_id,
        "dataset_sha256": record.dataset_sha256,
        "protocol_sha256": record.protocol_sha256,
    }


def _record_binding(record: CalibrationRecord) -> dict[str, str]:
    return {
        "biological_id": record.biological_id,
        "data_role": record.data_role,
        "dataset_id": record.dataset_id,
        "dataset_sha256": record.dataset_sha256,
        "manifest_sha256": record.manifest_sha256,
        "mechanism": record.mechanism,
        "namespace": record.namespace,
        "protocol_sha256": record.protocol_sha256,
        "technical_view": record.technical_view,
    }


def _verify_tracked_calibration_contract() -> None:
    path = Path(__file__).resolve().parents[1] / _CALIBRATION_CONTRACT_PATH
    try:
        contract_bytes = path.read_bytes()
    except OSError as exc:
        raise RuntimeError("tracked calibration retention contract is unavailable") from exc
    if hashlib.sha256(contract_bytes).hexdigest() != CALIBRATION_CONTRACT_SHA256:
        raise ValueError("tracked calibration retention contract digest differs")


def fit_development_calibration(
    records: object,
    thresholds: CalibrationThresholds = CalibrationThresholds(),
) -> CalibrationArtifact:
    """Evaluate, select, and fit the retained calibrator on all development data."""

    _verify_tracked_calibration_contract()
    thresholds = _validated_thresholds(thresholds)
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
    selected_cross_validation = cross_validate_calibrator(
        canonical,
        decision.selected_algorithm,
    )
    if selected_cross_validation.fit_failures:
        raise RuntimeError("selected calibrator contains a failed development fold")
    groups = []
    for mechanism, biological_id in sorted(
        {(record.mechanism, record.biological_id) for record in canonical}
    ):
        groups.append(
            {
                "namespace": _DEVELOPMENT_NAMESPACE,
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
        "schema_version": 3,
        "artifact_type": "maskimpute_prezero_calibration",
        "estimand": "pre_capture_zero_given_observed_zero",
        "inference_features": ["p_pre_zero"],
        "data_scope": {
            "allowed_biological_ids": list(_DEVELOPMENT_BIOLOGICAL_IDS),
            "data_role": _DEVELOPMENT_DATA_ROLE,
            "namespace": _DEVELOPMENT_NAMESPACE,
            "protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
        },
        "retention_contract": {
            "contract_id": _CALIBRATION_CONTRACT_ID,
            "path": _CALIBRATION_CONTRACT_PATH,
            "sha256": CALIBRATION_CONTRACT_SHA256,
        },
        "truth_eligibility": {
            "accepted_truth_kind": "exact_pre_capture",
            "eligible_mechanisms": ["symsim"],
            "eligible_mechanism_count": 1,
            "minimum_exact_mechanisms_improved": (
                thresholds.minimum_exact_mechanisms_improved
            ),
            "minimum_biological_draws_improved": (
                thresholds.minimum_biological_draws_improved
            ),
            "minimum_technical_records_improved": (
                thresholds.minimum_technical_records_improved
            ),
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
        "development_holdout_calibrators": [
            {
                "mechanism": fold.mechanism,
                "biological_id": fold.biological_id,
                "held_out_manifest_sha256s": sorted(fold.held_out_manifests),
                "training_manifest_sha256s": sorted(fold.training_manifests),
                "calibrator": fold.calibrator.to_dict(),
            }
            for fold in selected_cross_validation.folds
        ],
        "training": {
            "record_count": len(canonical),
            "entry_count": sum(len(record.p_pre_zero) for record in canonical),
            "record_digest_sha256": _canonical_digest(
                [_record_payload(record) for record in canonical]
            ),
            "record_bindings": [_record_binding(record) for record in canonical],
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


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if not os.path.lexists(component):
            continue
        try:
            metadata = component.lstat()
        except OSError as exc:
            raise ValueError(f"calibration path cannot be inspected: {path}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"calibration path must not contain symlinks: {path}")


def _file_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_unique_regular_bytes(path: Path, name: str) -> bytes:
    _reject_symlink_components(path)
    try:
        before_path = path.lstat()
        if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
            raise ValueError(f"{name} must be a unique regular file without links")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"{name} cannot be opened as a unique regular file") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise ValueError(f"{name} changed while opening or is linked")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        after_path = path.lstat()
        if _file_state(before) != _file_state(after) or _file_state(
            before
        ) != _file_state(after_path):
            raise ValueError(f"{name} changed while reading")
        _reject_symlink_components(path)
        return b"".join(chunks)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"{name} changed while reading") from exc
    finally:
        os.close(descriptor)


def _verify_open_directory(path: Path, descriptor: int) -> None:
    try:
        by_path = path.lstat()
        by_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("calibration output directory changed") from exc
    if (
        not stat.S_ISDIR(by_path.st_mode)
        or stat.S_ISLNK(by_path.st_mode)
        or not stat.S_ISDIR(by_descriptor.st_mode)
        or (by_path.st_dev, by_path.st_ino)
        != (by_descriptor.st_dev, by_descriptor.st_ino)
    ):
        raise ValueError("calibration output directory changed or is a symlink")


def load_calibration_artifact(path: str | Path) -> CalibrationArtifact:
    """Load exact canonical bytes and verify the complete artifact schema/digest."""

    artifact_path = Path(path)
    try:
        raw = _read_unique_regular_bytes(
            artifact_path,
            "calibration artifact",
        )
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
        raw = _read_unique_regular_bytes(
            input_path,
            "calibration training input",
        )
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
    if type(root["schema_version"]) is not int or root["schema_version"] != 2:
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
                "namespace",
                "data_role",
                "technical_view",
                "dataset_id",
                "dataset_sha256",
                "protocol_sha256",
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
    output = Path(path).absolute()
    _reject_symlink_components(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output)
    parent_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(output.parent, parent_flags)
    descriptor = -1
    temporary_name: str | None = None
    published = False
    publication_identity: tuple[int, int] | None = None
    temporary_removed = False
    try:
        _verify_open_directory(output.parent, parent_descriptor)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        temporary_path = Path(temporary_name)
        temporary_name = temporary_path.name
        destination = os.fdopen(descriptor, "wb")
        descriptor = -1
        with destination:
            destination.write(artifact._payload_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        temporary_metadata = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(temporary_metadata.st_mode):
            raise OSError("calibration temporary artifact is not a regular file")
        publication_identity = (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
        )
        _verify_open_directory(output.parent, parent_descriptor)
        _reject_symlink_components(output)
        os.link(
            temporary_name,
            output.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published = True
        _verify_open_directory(output.parent, parent_descriptor)
        os.fsync(parent_descriptor)
        _verify_open_directory(output.parent, parent_descriptor)
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_removed = True
        os.fsync(parent_descriptor)
        _verify_open_directory(output.parent, parent_descriptor)
        output_metadata = os.stat(
            output.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(output_metadata.st_mode)
            or output_metadata.st_nlink != 1
            or (output_metadata.st_dev, output_metadata.st_ino) != publication_identity
        ):
            raise OSError("published calibration artifact identity changed")
    except BaseException:
        if published and publication_identity is not None:
            try:
                output_metadata = os.stat(
                    output.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISREG(output_metadata.st_mode)
                    and (
                        output_metadata.st_dev,
                        output_metadata.st_ino,
                    )
                    == publication_identity
                ):
                    os.unlink(output.name, dir_fd=parent_descriptor)
                    try:
                        os.fsync(parent_descriptor)
                    except OSError:
                        pass
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None and not temporary_removed:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        os.close(parent_descriptor)
