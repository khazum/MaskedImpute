"""Public API for calibrated selective count imputation."""

from typing import TYPE_CHECKING

from .config import MaskImputeConfig
from .count_model import (
    PreZeroCountModelConfig,
    PreZeroCountModelScore,
    fit_p_pre_zero_count_model,
)
from .prezero import p_pre_zero_from_counts
from .result import ImputationResult

if TYPE_CHECKING:
    from .impute import impute_counts

__all__ = [
    "ImputationResult",
    "MaskImputeConfig",
    "PreZeroCountModelConfig",
    "PreZeroCountModelScore",
    "fit_p_pre_zero_count_model",
    "impute_counts",
    "p_pre_zero_from_counts",
]


def __getattr__(name: str):
    """Load the torch-backed API only when it is explicitly requested."""

    if name == "impute_counts":
        from .impute import impute_counts

        globals()[name] = impute_counts
        return impute_counts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
