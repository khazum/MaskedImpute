"""Training utilities for the leakage-safe MaskImpute v27 model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import copy
from functools import wraps
import hashlib
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import sparse

from maskimpute.sparse_input import (
    SUPPORTED_SPARSE_TYPES as _SUPPORTED_SPARSE_TYPES,
    contains_masked_array as _contains_masked_array,
    sparse_coordinate_snapshot,
)

if TYPE_CHECKING:
    from maskimpute.config import MaskImputeConfig


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Internal immutable audit record for one deterministic masked fit."""

    model: torch.nn.Module
    normalized_expression: np.ndarray
    library_sizes: np.ndarray
    validation_mask: np.ndarray
    training_loss_history: tuple[float, ...]
    validation_loss_history: tuple[float, ...]
    validation_mask_hashes: tuple[str, ...]
    epoch_training_mask_hashes: tuple[str, ...]
    best_epoch: int
    stopped_epoch: int
    validation_seed: int
    training_seed: int
    device: str
    deterministic_algorithms: bool
    caller_rng_state_restored: bool
    cublas_workspace_config: str | None


@dataclass(frozen=True, slots=True)
class V28TrainingOutcome:
    """NB training record plus the fixed gene-dispersion nuisance estimate."""

    training: TrainingOutcome
    dispersion: object

    def __post_init__(self) -> None:
        from maskimpute.nb_model import GeneDispersionEstimate

        if type(self.training) is not TrainingOutcome:
            raise TypeError("training must be an exact TrainingOutcome")
        if type(self.dispersion) is not GeneDispersionEstimate:
            raise TypeError("dispersion must be an exact GeneDispersionEstimate")


@dataclass(frozen=True, slots=True)
class V29TrainingOutcome:
    """NB training record with frozen observed-only structure authority."""

    training: TrainingOutcome
    dispersion: object
    structure: object

    def __post_init__(self) -> None:
        from maskimpute.nb_model import GeneDispersionEstimate
        from maskimpute.structure import StructureAuthority

        if type(self.training) is not TrainingOutcome:
            raise TypeError("training must be an exact TrainingOutcome")
        if type(self.dispersion) is not GeneDispersionEstimate:
            raise TypeError("dispersion must be an exact GeneDispersionEstimate")
        if type(self.structure) is not StructureAuthority:
            raise TypeError("structure must be an exact StructureAuthority")


_MAX_EXACT_FLOAT64_INTEGER = 2**53


def _unsummed_sparse_coordinates(value: object, name: str):
    return sparse_coordinate_snapshot(value, name)


def _numeric_matrix_to_dense(value: object, name: str) -> tuple[np.ndarray, np.ndarray]:
    is_sparse = sparse.issparse(value)
    if is_sparse and type(value) not in _SUPPORTED_SPARSE_TYPES:
        raise TypeError(f"{name} must use an exact supported SciPy sparse type")
    if is_sparse:
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if value.dtype.metadata is not None:
            raise TypeError(f"{name} dtype metadata is not supported")
        entries, rows, columns, shape = _unsummed_sparse_coordinates(value, name)
    else:
        if _contains_masked_array(value):
            raise TypeError(f"{name} must not contain masked arrays")
        coerced = np.asanyarray(value)
        if np.ma.isMaskedArray(coerced):
            raise TypeError(f"{name} must not contain masked arrays")
        matrix = np.asarray(coerced)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if matrix.dtype.metadata is not None:
            raise TypeError(f"{name} dtype metadata is not supported")
        entries = matrix
        shape = matrix.shape
    if 0 in shape:
        raise ValueError(f"{name} must have at least one cell and one gene")
    if entries.dtype.kind not in "iuf" or entries.dtype.kind == "b":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(entries)):
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        if is_sparse:
            dense = np.zeros(shape, dtype=np.float64, order="C")
            dense[rows, columns] = entries
        else:
            dense = np.array(
                matrix,
                dtype=np.float64,
                copy=True,
                order="C",
                subok=False,
            )
    if not np.all(np.isfinite(dense)):
        raise ValueError(f"{name} must be representable as finite float64 values")
    return dense, entries


def validate_observed_counts(observed_counts: object) -> np.ndarray:
    """Validate raw counts without silently coercing ambiguous input."""

    counts, original_entries = _numeric_matrix_to_dense(
        observed_counts,
        "observed_counts",
    )
    if np.any(original_entries < 0):
        raise ValueError("observed_counts must be nonnegative")
    if np.any(original_entries != np.floor(original_entries)):
        raise ValueError("observed_counts must contain integral counts")
    if np.any(original_entries > _MAX_EXACT_FLOAT64_INTEGER):
        raise ValueError("observed_counts exceed exact float64 count range")
    return counts


def validate_p_pre_zero(
    p_pre_zero: object,
    observed_counts: np.ndarray,
) -> np.ndarray:
    """Validate the external score and its observed-zero support."""

    probability, _ = _numeric_matrix_to_dense(p_pre_zero, "p_pre_zero")
    if probability.shape != observed_counts.shape:
        raise ValueError("p_pre_zero shape must match observed_counts")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("p_pre_zero must lie in [0, 1]")
    if np.any(probability[observed_counts > 0] != 0):
        raise ValueError("p_pre_zero must be zero at observed positive entries")
    return probability


def _finite_matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.dtype.kind not in "iuf" or matrix.dtype.kind == "b":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def normalize_observed_counts(
    observed_counts: object,
    *,
    target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``log1p(count / observed_library * target)``.

    Cells whose observed library is zero remain identically zero.  The paired
    :func:`invert_observed_normalization` function uses the same policy.
    """

    counts = validate_observed_counts(observed_counts)
    if isinstance(target, bool) or not np.isfinite(target) or target <= 0:
        raise ValueError("target must be positive and finite")
    target_float = float(target)
    if not np.isfinite(target_float) or target_float <= 0:
        raise ValueError("target must be positive and finite")
    library_sizes = counts.sum(axis=1, dtype=np.float64)
    np.divide(
        counts,
        library_sizes[:, None],
        out=counts,
        where=library_sizes[:, None] > 0,
    )
    np.multiply(counts, target_float, out=counts)
    np.log1p(counts, out=counts)
    return counts, library_sizes


def normalize_available_encoder_input(
    observed_counts: object,
    availability: object,
    *,
    target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize only the count payload visible in a corrupted encoder view."""

    counts = validate_observed_counts(observed_counts)
    if _contains_masked_array(availability):
        raise TypeError("availability must not contain masked arrays")
    coerced = np.asanyarray(availability)
    if np.ma.isMaskedArray(coerced):
        raise TypeError("availability must not contain masked arrays")
    if coerced.dtype.metadata is not None:
        raise TypeError("availability dtype metadata is not supported")
    available = np.array(coerced, copy=True, order="C", subok=False)
    if available.dtype != np.bool_ or available.shape != counts.shape:
        raise ValueError("availability must be boolean with the count-matrix shape")
    if isinstance(target, bool) or not np.isfinite(target) or target <= 0:
        raise ValueError("target must be positive and finite")
    target_float = float(target)
    if not np.isfinite(target_float) or target_float <= 0:
        raise ValueError("target must be positive and finite")

    np.multiply(counts, available, out=counts)
    visible_library_sizes = counts.sum(axis=1, dtype=np.float64)
    np.divide(
        counts,
        visible_library_sizes[:, None],
        out=counts,
        where=visible_library_sizes[:, None] > 0,
    )
    np.multiply(counts, target_float, out=counts)
    np.log1p(counts, out=counts)
    return counts, visible_library_sizes


def invert_observed_normalization(
    normalized_expression: object,
    library_sizes: object,
    *,
    target: float,
) -> np.ndarray:
    """Invert observed-library log normalization on count scale.

    A zero-library cell maps to an all-zero count vector because its count
    scale is undefined from the observed data; MaskImpute never invents a
    library-size offset for such cells.
    """

    normalized = _finite_matrix(normalized_expression, "normalized_expression")
    if np.ma.isMaskedArray(library_sizes):
        raise TypeError("library_sizes must not be a masked array")
    libraries = np.asarray(library_sizes)
    if libraries.ndim != 1 or libraries.shape[0] != normalized.shape[0]:
        raise ValueError("library_sizes must have one entry per cell")
    if libraries.dtype.kind not in "iuf" or libraries.dtype.kind == "b":
        raise TypeError("library_sizes must contain real numeric values")
    libraries = np.asarray(libraries, dtype=np.float64)
    if not np.all(np.isfinite(libraries)) or np.any(libraries < 0):
        raise ValueError("library_sizes must be finite and nonnegative")
    if isinstance(target, bool) or not np.isfinite(target) or target <= 0:
        raise ValueError("target must be positive and finite")
    target_float = float(target)
    if not np.isfinite(target_float) or target_float <= 0:
        raise ValueError("target must be positive and finite")

    restored = np.expm1(normalized) * (libraries[:, None] / target_float)
    restored[libraries == 0] = 0.0
    return restored


def _validated_edges(log_count_bin_edges: Sequence[float]) -> np.ndarray:
    edges = np.asarray(tuple(log_count_bin_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size == 0:
        raise ValueError("log_count_bin_edges must be a nonempty sequence")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("log_count_bin_edges must be finite and strictly increasing")
    return edges


def make_stratified_validation_mask(
    observed_counts: object,
    *,
    fraction: float,
    log_count_bin_edges: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a fixed positive holdout, stratified by prespecified log bins."""

    counts = validate_observed_counts(observed_counts)
    holdout_fraction = _fraction(fraction, "fraction")
    edges = _validated_edges(log_count_bin_edges)
    positive_flat = np.flatnonzero(counts.ravel() > 0)
    if positive_flat.size < 2:
        raise ValueError("at least two observed positive entries are required")

    positive_values = counts.ravel()[positive_flat]
    strata = np.searchsorted(edges, np.log1p(positive_values), side="right")
    target = max(
        1,
        min(
            positive_flat.size - 1,
            int(np.floor(positive_flat.size * holdout_fraction + 0.5)),
        ),
    )

    stratum_ids = np.unique(strata)
    sizes = np.array([np.count_nonzero(strata == item) for item in stratum_ids])
    ideal = sizes.astype(np.float64) * holdout_fraction
    allocation = np.minimum(np.floor(ideal).astype(int), np.maximum(sizes - 1, 0))

    remaining = target - int(allocation.sum())
    if remaining > 0:
        tie_order = rng.permutation(len(stratum_ids))
        ranked = sorted(
            tie_order.tolist(),
            key=lambda index: ideal[index] - np.floor(ideal[index]),
            reverse=True,
        )
        while remaining > 0:
            changed = False
            for index in ranked:
                capacity = max(int(sizes[index]) - 1, 0)
                if allocation[index] < capacity:
                    allocation[index] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

    chosen: list[int] = []
    for index, stratum in enumerate(stratum_ids):
        members = positive_flat[strata == stratum]
        count = int(allocation[index])
        if count:
            chosen.extend(rng.choice(members, size=count, replace=False).tolist())

    if len(chosen) < target:
        available = np.setdiff1d(
            positive_flat,
            np.asarray(chosen, dtype=np.int64),
            assume_unique=False,
        )
        extra = rng.choice(available, size=target - len(chosen), replace=False)
        chosen.extend(extra.tolist())

    mask = np.zeros(counts.size, dtype=np.bool_)
    mask[np.asarray(chosen, dtype=np.int64)] = True
    return mask.reshape(counts.shape)


def make_epoch_training_mask(
    observed_counts: object,
    *,
    validation_mask: object,
    fraction: float,
    log_count_bin_edges: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Mask only observed positives not reserved for fixed validation."""

    counts = validate_observed_counts(observed_counts)
    validation = np.asarray(validation_mask)
    if validation.dtype != np.bool_ or validation.shape != counts.shape:
        raise ValueError("validation_mask must be boolean with the count-matrix shape")
    mask_fraction = _fraction(fraction, "fraction")
    edges = _validated_edges(log_count_bin_edges)
    candidates = np.flatnonzero(((counts > 0) & ~validation).ravel())
    if candidates.size == 0:
        raise ValueError("no training positives remain outside validation")
    number = max(
        1,
        min(
            candidates.size,
            int(np.floor(candidates.size * mask_fraction + 0.5)),
        ),
    )

    candidate_values = counts.ravel()[candidates]
    strata = np.searchsorted(edges, np.log1p(candidate_values), side="right")
    stratum_ids = np.unique(strata)
    sizes = np.array([np.count_nonzero(strata == item) for item in stratum_ids])
    ideal = sizes.astype(np.float64) * mask_fraction
    allocation = np.floor(ideal).astype(int)
    remaining = number - int(allocation.sum())
    if remaining > 0:
        tie_order = rng.permutation(len(stratum_ids))
        ranked = sorted(
            tie_order.tolist(),
            key=lambda index: ideal[index] - np.floor(ideal[index]),
            reverse=True,
        )
        while remaining > 0:
            for index in ranked:
                if allocation[index] < sizes[index]:
                    allocation[index] += 1
                    remaining -= 1
                    if remaining == 0:
                        break

    chosen_items: list[int] = []
    for index, stratum in enumerate(stratum_ids):
        members = candidates[strata == stratum]
        count = int(allocation[index])
        if count:
            chosen_items.extend(rng.choice(members, size=count, replace=False).tolist())

    result = np.zeros(counts.size, dtype=np.bool_)
    result[np.asarray(chosen_items, dtype=np.int64)] = True
    return result.reshape(counts.shape)


def natural_zero_preservation_loss(
    predictions: torch.Tensor,
    natural_zero_mask: torch.Tensor,
    p_pre_zero: torch.Tensor,
) -> torch.Tensor:
    """Return the soft external-score-weighted zero preservation penalty."""

    if (
        predictions.shape != natural_zero_mask.shape
        or predictions.shape != p_pre_zero.shape
    ):
        raise ValueError("prediction, natural-zero, and probability shapes must match")
    if natural_zero_mask.dtype != torch.bool:
        raise TypeError("natural_zero_mask must be boolean")
    if not torch.any(natural_zero_mask):
        return predictions.sum() * 0.0
    weighted_squared = (
        p_pre_zero[natural_zero_mask] * predictions[natural_zero_mask].square()
    )
    return weighted_squared.mean()


def _masked_positive_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    artificial_mask: torch.Tensor,
) -> torch.Tensor:
    if torch.any(artificial_mask):
        return (predictions[artificial_mask] - targets[artificial_mask]).square().mean()
    return predictions.sum() * 0.0


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


@contextmanager
def _scoped_deterministic_torch(seed: int, device: torch.device):
    previous_cpu_rng = torch.random.get_rng_state().clone()
    previous_cuda_rng = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if device.type == "cuda"
        else None
    )
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(previous_cpu_rng)
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )


def _deterministic_training_scope(function):
    @wraps(function)
    def wrapped(
        observed_counts: object,
        p_pre_zero: object,
        config: MaskImputeConfig,
        device: str | torch.device,
        *args: object,
        **kwargs: object,
    ) -> TrainingOutcome:
        selected_device = torch.device(device)
        if selected_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace not in {":4096:8", ":16:8"}:
                raise RuntimeError(
                    "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG "
                    "to be ':4096:8' or ':16:8'"
                )
        with _scoped_deterministic_torch(config.seed, selected_device):
            return function(
                observed_counts,
                p_pre_zero,
                config,
                selected_device,
                *args,
                **kwargs,
            )

    return wrapped


@_deterministic_training_scope
def _train_with_policies(
    observed_counts: object,
    p_pre_zero: object,
    config: MaskImputeConfig,
    device: str | torch.device,
    *,
    model_factory: Callable[[int, MaskImputeConfig], torch.nn.Module],
    training_mask_factory: Callable[..., np.ndarray],
    objective_factory: Callable[..., Callable[..., tuple[torch.Tensor, torch.Tensor]]]
    | None = None,
    additional_training_loss: Callable[
        [torch.Tensor, torch.Tensor, np.ndarray], torch.Tensor
    ]
    | None = None,
) -> TrainingOutcome:
    """Shared deterministic trainer for prespecified architecture/mask policies."""

    counts = validate_observed_counts(observed_counts)
    probability = validate_p_pre_zero(p_pre_zero, counts)

    normalized, library_sizes = normalize_observed_counts(
        counts,
        target=config.normalization_target,
    )
    seed_sequence = np.random.SeedSequence(config.seed)
    validation_sequence, training_sequence = seed_sequence.spawn(2)
    validation_seed = int(validation_sequence.generate_state(1, dtype=np.uint64)[0])
    training_seed = int(training_sequence.generate_state(1, dtype=np.uint64)[0])
    validation_rng = np.random.default_rng(validation_seed)
    training_rng = np.random.default_rng(training_seed)
    validation_mask = make_stratified_validation_mask(
        counts,
        fraction=config.validation_fraction,
        log_count_bin_edges=config.log_count_bin_edges,
        rng=validation_rng,
    )
    fixed_validation_hash = _mask_hash(validation_mask)

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = model_factory(counts.shape[1], config)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    model = model.to(selected_device)
    if not tuple(model.parameters()):
        raise ValueError("training model must contain parameters")
    if objective_factory is not None and not callable(objective_factory):
        raise TypeError("objective_factory must be callable or None")
    if additional_training_loss is not None and not callable(additional_training_loss):
        raise TypeError("additional_training_loss must be callable or None")
    objective = (
        None
        if objective_factory is None
        else objective_factory(
            counts,
            library_sizes,
            validation_mask,
            config,
            selected_device,
        )
    )
    if objective is not None and not callable(objective):
        raise TypeError("objective_factory must return a callable objective")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    target_expression = torch.as_tensor(
        normalized, dtype=torch.float32, device=selected_device
    )
    target_counts = None
    count_library_sizes = None
    if objective is not None:
        target_counts = torch.as_tensor(
            counts,
            dtype=torch.float64,
            device=selected_device,
        )
        count_library_sizes = torch.as_tensor(
            library_sizes,
            dtype=torch.float64,
            device=selected_device,
        )
    positive = torch.as_tensor(counts > 0, dtype=torch.bool, device=selected_device)
    natural_zero = ~positive
    score = torch.as_tensor(probability, dtype=torch.float32, device=selected_device)
    fixed_validation = torch.as_tensor(
        validation_mask,
        dtype=torch.bool,
        device=selected_device,
    )
    base_availability_numpy = (counts > 0) & ~validation_mask
    validation_input_numpy, _ = normalize_available_encoder_input(
        counts,
        base_availability_numpy,
        target=config.normalization_target,
    )
    validation_input = torch.as_tensor(
        validation_input_numpy,
        dtype=torch.float32,
        device=selected_device,
    )
    base_availability = torch.as_tensor(
        base_availability_numpy,
        dtype=torch.bool,
        device=selected_device,
    )

    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_hashes: list[str] = []
    training_mask_hashes: list[str] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch_index in range(config.max_epochs):
        model.train()
        epoch_mask = training_mask_factory(
            counts,
            validation_mask=validation_mask,
            fraction=config.artificial_mask_fraction,
            log_count_bin_edges=config.log_count_bin_edges,
            rng=training_rng,
        )
        training_mask_hashes.append(_mask_hash(epoch_mask))
        artificial = torch.as_tensor(
            epoch_mask,
            dtype=torch.bool,
            device=selected_device,
        )
        corrupted_availability_numpy = base_availability_numpy & ~epoch_mask
        corrupted_input_numpy, _ = normalize_available_encoder_input(
            counts,
            corrupted_availability_numpy,
            target=config.normalization_target,
        )
        corrupted_input = torch.as_tensor(
            corrupted_input_numpy,
            dtype=torch.float32,
            device=selected_device,
        )
        corrupted_availability = torch.as_tensor(
            corrupted_availability_numpy,
            dtype=torch.bool,
            device=selected_device,
        )

        row_order = training_rng.permutation(counts.shape[0])
        batch_losses: list[float] = []
        for start in range(0, counts.shape[0], config.batch_size):
            rows_numpy = row_order[start : start + config.batch_size]
            rows = torch.as_tensor(rows_numpy, dtype=torch.long, device=selected_device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(
                corrupted_input[rows],
                corrupted_availability[rows],
            )
            if objective is None:
                primary_loss = _masked_positive_loss(
                    prediction,
                    target_expression[rows],
                    artificial[rows],
                )
                preservation_loss = natural_zero_preservation_loss(
                    prediction,
                    natural_zero[rows],
                    score[rows],
                )
            else:
                assert target_counts is not None and count_library_sizes is not None
                primary_loss, preservation_loss = objective(
                    prediction,
                    target_counts[rows],
                    count_library_sizes[rows],
                    artificial[rows],
                    natural_zero[rows],
                    score[rows],
                )
            loss = primary_loss + config.pre_zero_regularization * preservation_loss
            if additional_training_loss is not None:
                additional = additional_training_loss(
                    prediction,
                    corrupted_input[rows],
                    rows_numpy,
                )
                if not isinstance(additional, torch.Tensor) or additional.ndim != 0:
                    raise TypeError(
                        "additional_training_loss must return a scalar tensor"
                    )
                loss = loss + additional
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite masked-model training loss")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        training_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(
                validation_input,
                base_availability,
            )
            if objective is None:
                validation_loss = _masked_positive_loss(
                    validation_prediction,
                    target_expression,
                    fixed_validation,
                )
            else:
                assert target_counts is not None and count_library_sizes is not None
                validation_loss, _ = objective(
                    validation_prediction,
                    target_counts,
                    count_library_sizes,
                    fixed_validation,
                    natural_zero,
                    score,
                )
        validation_value = float(validation_loss.detach().cpu())
        if not np.isfinite(validation_value):
            raise FloatingPointError("nonfinite masked-model validation loss")
        validation_losses.append(validation_value)
        validation_hashes.append(fixed_validation_hash)

        if validation_value < best_loss - config.early_stopping_min_delta:
            best_loss = validation_value
            best_epoch = epoch_index + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("masked-model training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    return TrainingOutcome(
        model=model,
        normalized_expression=np.array(normalized, copy=True),
        library_sizes=np.array(library_sizes, copy=True),
        validation_mask=np.array(validation_mask, copy=True),
        training_loss_history=tuple(training_losses),
        validation_loss_history=tuple(validation_losses),
        validation_mask_hashes=tuple(validation_hashes),
        epoch_training_mask_hashes=tuple(training_mask_hashes),
        best_epoch=best_epoch,
        stopped_epoch=len(validation_losses),
        validation_seed=validation_seed,
        training_seed=training_seed,
        device=str(selected_device),
        deterministic_algorithms=True,
        caller_rng_state_restored=True,
        cublas_workspace_config=(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if selected_device.type == "cuda"
            else None
        ),
    )


def train_v27(
    observed_counts: object,
    p_pre_zero: object,
    config: MaskImputeConfig,
    device: str | torch.device,
) -> TrainingOutcome:
    """Train v27 using only observed counts and the external count score."""

    from maskimpute.model import ExplicitMaskAutoencoder

    def model_factory(
        n_genes: int,
        training_config: MaskImputeConfig,
    ) -> torch.nn.Module:
        return ExplicitMaskAutoencoder(
            n_genes=n_genes,
            hidden_dims=training_config.hidden_dims,
            latent_dim=training_config.latent_dim,
        )

    return _train_with_policies(
        observed_counts,
        p_pre_zero,
        config,
        device,
        model_factory=model_factory,
        training_mask_factory=make_epoch_training_mask,
    )


def train_v28(
    observed_counts: object,
    p_pre_zero: object,
    config: MaskImputeConfig,
    device: str | torch.device,
    *,
    decoder_config: object,
) -> V28TrainingOutcome:
    """Train the conditional NB decoder from counts and an external score only."""

    from maskimpute.nb_model import (
        MAX_V28_COUNT_OR_LIBRARY,
        NegativeBinomialDecoderConfig,
        NegativeBinomialMaskAutoencoder,
        _negative_binomial_objective,
        estimate_shrunk_gene_dispersion,
    )

    if type(decoder_config) is not NegativeBinomialDecoderConfig:
        raise TypeError("decoder_config must be an exact NegativeBinomialDecoderConfig")
    counts = validate_observed_counts(observed_counts)
    library_sizes = counts.sum(axis=1, dtype=np.float64)
    if np.any(counts > MAX_V28_COUNT_OR_LIBRARY) or np.any(
        library_sizes > MAX_V28_COUNT_OR_LIBRARY
    ):
        raise ValueError(
            "v28 observed count or library exceeds the stable likelihood limit"
        )
    dispersion_holder: dict[str, object] = {}

    def model_factory(
        n_genes: int,
        training_config: MaskImputeConfig,
    ) -> torch.nn.Module:
        return NegativeBinomialMaskAutoencoder(
            n_genes=n_genes,
            hidden_dims=training_config.hidden_dims,
            latent_dim=training_config.latent_dim,
        )

    def objective_factory(
        counts: np.ndarray,
        library_sizes: np.ndarray,
        validation_mask: np.ndarray,
        training_config: MaskImputeConfig,
        selected_device: torch.device,
    ):
        dispersion = estimate_shrunk_gene_dispersion(
            counts,
            library_sizes,
            decoder_config,
            estimation_mask=~validation_mask,
        )
        dispersion_holder["value"] = dispersion
        return _negative_binomial_objective(
            dispersion,
            normalization_target=training_config.normalization_target,
            device=selected_device,
            dtype=torch.float64,
        )

    training = _train_with_policies(
        counts,
        p_pre_zero,
        config,
        device,
        model_factory=model_factory,
        training_mask_factory=make_epoch_training_mask,
        objective_factory=objective_factory,
    )
    dispersion = dispersion_holder.get("value")
    return V28TrainingOutcome(training=training, dispersion=dispersion)


def train_v29(
    observed_counts: object,
    p_pre_zero: object,
    config: MaskImputeConfig,
    device: str | torch.device,
    *,
    decoder_config: object,
    structure_config: object,
) -> V29TrainingOutcome:
    """Train the conditional v29 NB model with observed-only structure losses."""

    from maskimpute.nb_model import (
        MAX_V28_COUNT_OR_LIBRARY,
        NegativeBinomialDecoderConfig,
        NegativeBinomialMaskAutoencoder,
        _negative_binomial_objective,
        apply_library_size_offset,
        estimate_shrunk_gene_dispersion,
    )
    from maskimpute.structure import (
        StructurePenaltyConfig,
        build_structure_authority,
        structure_preservation_loss,
    )

    if type(decoder_config) is not NegativeBinomialDecoderConfig:
        raise TypeError("decoder_config must be an exact NegativeBinomialDecoderConfig")
    if type(structure_config) is not StructurePenaltyConfig:
        raise TypeError("structure_config must be an exact StructurePenaltyConfig")
    counts = validate_observed_counts(observed_counts)
    library_sizes = counts.sum(axis=1, dtype=np.float64)
    if np.any(counts > MAX_V28_COUNT_OR_LIBRARY) or np.any(
        library_sizes > MAX_V28_COUNT_OR_LIBRARY
    ):
        raise ValueError(
            "v29 observed count or library exceeds the stable likelihood limit"
        )
    dispersion_holder: dict[str, object] = {}
    structure_holder: dict[str, object] = {}

    def model_factory(
        n_genes: int,
        training_config: MaskImputeConfig,
    ) -> torch.nn.Module:
        return NegativeBinomialMaskAutoencoder(
            n_genes=n_genes,
            hidden_dims=training_config.hidden_dims,
            latent_dim=training_config.latent_dim,
        )

    def objective_factory(
        counts: np.ndarray,
        library_sizes: np.ndarray,
        validation_mask: np.ndarray,
        training_config: MaskImputeConfig,
        selected_device: torch.device,
    ):
        structure_counts = counts.copy()
        structure_counts[validation_mask] = 0.0
        structure_holder["value"] = build_structure_authority(
            structure_counts,
            structure_config,
        )
        dispersion = estimate_shrunk_gene_dispersion(
            counts,
            library_sizes,
            decoder_config,
            estimation_mask=~validation_mask,
        )
        dispersion_holder["value"] = dispersion
        return _negative_binomial_objective(
            dispersion,
            normalization_target=training_config.normalization_target,
            device=selected_device,
            dtype=torch.float64,
        )

    libraries = np.asarray(library_sizes, dtype=np.float64)

    def additional_training_loss(
        prediction: torch.Tensor,
        visible_expression: torch.Tensor,
        global_rows: np.ndarray,
    ) -> torch.Tensor:
        row_libraries = torch.as_tensor(
            libraries[global_rows],
            dtype=torch.float64,
            device=prediction.device,
        )
        predicted_counts = apply_library_size_offset(
            prediction.to(torch.float64), row_libraries
        )
        scale = torch.zeros_like(row_libraries)
        positive = row_libraries > 0
        scale[positive] = config.normalization_target / row_libraries[positive]
        predicted_normalized = torch.log1p(predicted_counts * scale[:, None])
        structure = structure_holder.get("value")
        loss, _components = structure_preservation_loss(
            predicted_normalized,
            visible_expression.to(torch.float64),
            global_rows,
            structure,
            structure_config,
        )
        return loss

    training = _train_with_policies(
        counts,
        p_pre_zero,
        config,
        device,
        model_factory=model_factory,
        training_mask_factory=make_epoch_training_mask,
        objective_factory=objective_factory,
        additional_training_loss=additional_training_loss,
    )
    dispersion = dispersion_holder.get("value")
    structure = structure_holder.get("value")
    return V29TrainingOutcome(
        training=training,
        dispersion=dispersion,
        structure=structure,
    )
