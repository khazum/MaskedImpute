"""Immutable configuration for MaskImpute models."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from collections.abc import Sequence


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _finite_float(value: object, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier} and finite")
    return result


def _hidden_dims(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_dims must be a sequence of positive integers")
    result = tuple(_positive_int(item, "hidden_dims entry") for item in value)
    if not result:
        raise ValueError("hidden_dims must not be empty")
    return result


@dataclass(frozen=True, slots=True)
class MaskImputeConfig:
    """Configuration shared by MaskImpute training and inference."""

    hidden_dims: tuple[int, ...] = (128, 64)
    latent_dim: int = 24
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 30
    artificial_mask_fraction: float = 0.20
    pre_zero_regularization: float = 1.0
    gate_gamma: float = 1.0
    normalization_target: float = 10_000.0
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", _hidden_dims(self.hidden_dims))
        for name in ("latent_dim", "batch_size", "max_epochs", "patience"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))

        for name in ("learning_rate", "normalization_target"):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, positive=True),
            )
        for name in ("weight_decay", "pre_zero_regularization", "gate_gamma"):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, positive=False),
            )

        fraction = _finite_float(
            self.artificial_mask_fraction,
            "artificial_mask_fraction",
            positive=True,
        )
        if fraction >= 1:
            raise ValueError("artificial_mask_fraction must be less than 1")
        object.__setattr__(self, "artificial_mask_fraction", fraction)
