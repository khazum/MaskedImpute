"""Training utilities for the leakage-safe MaskImpute v27 model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

if TYPE_CHECKING:
    from maskimpute.config import MaskImputeConfig

    from maskimpute.model import ExplicitMaskAutoencoder


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Internal immutable audit record for one v27 fit."""

    model: ExplicitMaskAutoencoder
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


_MAX_EXACT_FLOAT64_INTEGER = 2**53


def _contains_masked_array(value: object, seen: set[int] | None = None) -> bool:
    if np.ma.isMaskedArray(value):
        return True
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if sparse.issparse(value):
        containers = []
        if hasattr(value, "data"):
            containers.append(value.data)
        if hasattr(value, "_dict"):
            containers.append(value._dict)
        return any(_contains_masked_array(item, seen) for item in containers)
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        return any(_contains_masked_array(item, seen) for item in value.flat)
    if isinstance(value, Mapping):
        return any(_contains_masked_array(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_masked_array(item, seen) for item in value)
    return False


def _unsummed_sparse_coordinates(value: object, name: str):
    coordinates = value.tocoo(copy=True)
    if coordinates.nnz < 2:
        return coordinates
    order = np.lexsort((coordinates.col, coordinates.row))
    rows = coordinates.row[order]
    columns = coordinates.col[order]
    if np.any((rows[1:] == rows[:-1]) & (columns[1:] == columns[:-1])):
        raise ValueError(f"{name} must not contain duplicate sparse coordinates")
    return coordinates


def _numeric_matrix_to_dense(value: object, name: str) -> tuple[np.ndarray, np.ndarray]:
    if _contains_masked_array(value):
        raise TypeError(f"{name} must not contain masked arrays")
    if sparse.issparse(value):
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if value.dtype.metadata is not None:
            raise TypeError(f"{name} dtype metadata is not supported")
        coordinates = _unsummed_sparse_coordinates(value, name)
        entries = np.asarray(coordinates.data)
        shape = value.shape
    else:
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
    if sparse.issparse(value):
        dense = np.asarray(coordinates.toarray(), dtype=np.float64, order="C")
    else:
        dense = np.array(matrix, dtype=np.float64, copy=True, order="C", subok=False)
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
    library_sizes = counts.sum(axis=1, dtype=np.float64)
    scale = np.zeros_like(library_sizes)
    np.divide(float(target), library_sizes, out=scale, where=library_sizes > 0)
    normalized = np.log1p(counts * scale[:, None])
    return normalized, library_sizes


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

    visible_counts = np.where(available, counts, 0.0)
    visible_library_sizes = visible_counts.sum(axis=1, dtype=np.float64)
    scale = np.zeros_like(visible_library_sizes)
    np.divide(
        float(target),
        visible_library_sizes,
        out=scale,
        where=visible_library_sizes > 0,
    )
    normalized = np.log1p(visible_counts * scale[:, None])
    return normalized, visible_library_sizes


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

    restored = np.expm1(normalized) * (libraries[:, None] / float(target))
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
            return function(observed_counts, p_pre_zero, config, selected_device)

    return wrapped


@_deterministic_training_scope
def train_v27(
    observed_counts: object,
    p_pre_zero: object,
    config: MaskImputeConfig,
    device: str | torch.device,
) -> TrainingOutcome:
    """Train v27 using only observed counts and the external count score."""

    from maskimpute.model import ExplicitMaskAutoencoder

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
    model = ExplicitMaskAutoencoder(
        n_genes=counts.shape[1],
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    target_expression = torch.as_tensor(
        normalized, dtype=torch.float32, device=selected_device
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
        epoch_mask = make_epoch_training_mask(
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
            loss = primary_loss + config.pre_zero_regularization * preservation_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite v27 training loss")
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
            validation_loss = _masked_positive_loss(
                validation_prediction,
                target_expression,
                fixed_validation,
            )
        validation_value = float(validation_loss.detach().cpu())
        if not np.isfinite(validation_value):
            raise FloatingPointError("nonfinite v27 validation loss")
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
        raise RuntimeError("v27 training produced no valid checkpoint")
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
