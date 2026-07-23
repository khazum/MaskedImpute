"""Negative-binomial decoder primitives for the conditional v28 candidate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import torch
from torch import nn


MAX_V28_COUNT_OR_LIBRARY = 10_000_000.0
MIN_V28_INVERSE_DISPERSION = 1e-2
MAX_V28_INVERSE_DISPERSION = 1e4


def _finite_real(
    value: object,
    name: str,
    *,
    positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class NegativeBinomialDecoderConfig:
    """Fixed nuisance-estimation policy for the v28 count decoder."""

    dispersion_prior_strength: float = 20.0
    winsor_quantile: float = 0.95
    min_dispersion: float = 1e-4
    max_dispersion: float = 100.0
    mean_floor: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dispersion_prior_strength",
            _finite_real(
                self.dispersion_prior_strength,
                "dispersion prior strength",
                positive=False,
            ),
        )
        quantile = _finite_real(
            self.winsor_quantile,
            "winsor_quantile",
            positive=True,
        )
        if not 0.5 <= quantile <= 1.0:
            raise ValueError("winsor_quantile must lie in [0.5, 1]")
        object.__setattr__(self, "winsor_quantile", quantile)
        minimum = _finite_real(
            self.min_dispersion,
            "min_dispersion",
            positive=True,
        )
        maximum = _finite_real(
            self.max_dispersion,
            "max_dispersion",
            positive=True,
        )
        if maximum <= minimum:
            raise ValueError("max_dispersion must exceed min_dispersion")
        if minimum < 1.0 / MAX_V28_INVERSE_DISPERSION:
            raise ValueError("min_dispersion is below the stable v28 bound")
        if maximum > 1.0 / MIN_V28_INVERSE_DISPERSION:
            raise ValueError("max_dispersion exceeds the stable v28 bound")
        object.__setattr__(self, "min_dispersion", minimum)
        object.__setattr__(self, "max_dispersion", maximum)
        object.__setattr__(
            self,
            "mean_floor",
            _finite_real(self.mean_floor, "mean_floor", positive=True),
        )

    def to_dict(self) -> dict[str, float]:
        """Return the exact canonical-JSON-compatible configuration."""

        return {
            "dispersion_prior_strength": self.dispersion_prior_strength,
            "winsor_quantile": self.winsor_quantile,
            "min_dispersion": self.min_dispersion,
            "max_dispersion": self.max_dispersion,
            "mean_floor": self.mean_floor,
        }


@dataclass(frozen=True, slots=True)
class GeneDispersionEstimate:
    """Auditable NB2 gene dispersion and inverse-dispersion estimates."""

    dispersion: np.ndarray
    inverse_dispersion: np.ndarray
    raw_dispersion: np.ndarray
    global_dispersion: float
    effective_observations: np.ndarray

    def __post_init__(self) -> None:
        dispersion = np.array(self.dispersion, dtype=np.float64, copy=True)
        inverse = np.array(self.inverse_dispersion, dtype=np.float64, copy=True)
        raw = np.array(self.raw_dispersion, dtype=np.float64, copy=True)
        effective = np.array(
            self.effective_observations,
            dtype=np.int64,
            copy=True,
        )
        if (
            dispersion.ndim != 1
            or inverse.shape != dispersion.shape
            or raw.shape != dispersion.shape
            or effective.shape != dispersion.shape
        ):
            raise ValueError("gene dispersion arrays must be aligned vectors")
        if (
            not np.all(np.isfinite(dispersion))
            or not np.all(np.isfinite(inverse))
            or np.any(dispersion <= 0)
            or np.any(inverse <= 0)
        ):
            raise ValueError("gene dispersion estimates must be finite and positive")
        if not math.isfinite(self.global_dispersion) or self.global_dispersion <= 0:
            raise ValueError("global dispersion must be finite and positive")
        if np.any(effective < 0):
            raise ValueError("effective observations must be nonnegative")
        for array in (dispersion, inverse, raw, effective):
            array.setflags(write=False)
        object.__setattr__(self, "dispersion", dispersion)
        object.__setattr__(self, "inverse_dispersion", inverse)
        object.__setattr__(self, "raw_dispersion", raw)
        object.__setattr__(self, "effective_observations", effective)
        object.__setattr__(self, "global_dispersion", float(self.global_dispersion))


def _estimation_mask(value: object | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=np.bool_)
    if np.ma.isMaskedArray(value):
        raise TypeError("estimation_mask must not be a masked array")
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError("estimation_mask must be boolean with the count shape")
    return np.array(mask, copy=True, order="C")


def estimate_shrunk_gene_dispersion(
    observed_counts: object,
    library_sizes: object,
    config: NegativeBinomialDecoderConfig = NegativeBinomialDecoderConfig(),
    *,
    estimation_mask: object | None = None,
) -> GeneDispersionEstimate:
    """Estimate robust, exposure-adjusted, log-shrunk NB2 dispersions.

    The NB2 convention is ``variance = mean + dispersion * mean**2``.  The
    returned ``inverse_dispersion`` is therefore the SciPy/Torch NB size.
    """

    from maskimpute.train import validate_observed_counts

    if type(config) is not NegativeBinomialDecoderConfig:
        raise TypeError("config must be an exact NegativeBinomialDecoderConfig")
    counts = validate_observed_counts(observed_counts)
    libraries = np.asarray(library_sizes)
    if (
        libraries.ndim != 1
        or libraries.shape[0] != counts.shape[0]
        or libraries.dtype.kind not in "iuf"
        or libraries.dtype.kind == "b"
    ):
        raise ValueError(
            "library_sizes must be a numeric vector with one value per cell"
        )
    libraries = np.asarray(libraries, dtype=np.float64)
    if not np.all(np.isfinite(libraries)) or np.any(libraries < 0):
        raise ValueError("library_sizes must be finite and nonnegative")
    if not np.array_equal(libraries, counts.sum(axis=1, dtype=np.float64)):
        raise ValueError("library_sizes must equal the observed row sums")
    positive_libraries = libraries[libraries > 0]
    if not positive_libraries.size:
        raise ValueError("at least one positive observed library is required")
    mask = _estimation_mask(estimation_mask, counts.shape)
    median_library = float(np.median(positive_libraries))
    exposure = libraries / median_library

    n_genes = counts.shape[1]
    raw = np.full(n_genes, np.nan, dtype=np.float64)
    effective = np.zeros(n_genes, dtype=np.int64)
    for gene in range(n_genes):
        valid = mask[:, gene] & (libraries > 0)
        effective[gene] = int(np.count_nonzero(valid))
        if effective[gene] < 2:
            continue
        gene_exposure = exposure[valid]
        denominator_exposure = float(gene_exposure.sum())
        if denominator_exposure <= 0:
            continue
        values = counts[valid, gene]
        rate = float(values.sum()) / denominator_exposure
        mean = gene_exposure * rate
        squared_mean = mean * mean
        informative = squared_mean > config.mean_floor**2
        if not np.any(informative):
            continue
        values = values[informative]
        mean = mean[informative]
        squared_mean = squared_mean[informative]
        contribution = ((values - mean) ** 2 - values) / np.maximum(
            squared_mean,
            config.mean_floor**2,
        )
        lower = float(np.quantile(contribution, 1.0 - config.winsor_quantile))
        upper = float(np.quantile(contribution, config.winsor_quantile))
        winsorized = np.clip(contribution, lower, upper)
        estimate = float(np.sum(winsorized * squared_mean) / np.sum(squared_mean))
        raw[gene] = max(estimate, config.min_dispersion)

    valid_raw = raw[np.isfinite(raw) & (raw > 0)]
    global_dispersion = (
        float(np.median(valid_raw)) if valid_raw.size else float(config.min_dispersion)
    )
    global_dispersion = float(
        np.clip(
            global_dispersion,
            config.min_dispersion,
            config.max_dispersion,
        )
    )
    usable_raw = np.where(np.isfinite(raw) & (raw > 0), raw, global_dispersion)
    usable_raw = np.clip(
        usable_raw,
        config.min_dispersion,
        config.max_dispersion,
    )
    weight = effective.astype(np.float64) / (
        effective.astype(np.float64) + config.dispersion_prior_strength
    )
    if config.dispersion_prior_strength == 0:
        weight = np.where(effective > 0, 1.0, 0.0)
    shrunk = np.exp(
        weight * np.log(usable_raw) + (1.0 - weight) * math.log(global_dispersion)
    )
    shrunk = np.clip(
        shrunk,
        config.min_dispersion,
        config.max_dispersion,
    )
    return GeneDispersionEstimate(
        dispersion=shrunk,
        inverse_dispersion=1.0 / shrunk,
        raw_dispersion=raw,
        global_dispersion=global_dispersion,
        effective_observations=effective,
    )


def negative_binomial_nll(
    counts: torch.Tensor,
    mean: torch.Tensor,
    inverse_dispersion: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return exact NB2 negative log likelihood in mean/size form."""

    if not isinstance(counts, torch.Tensor) or not isinstance(mean, torch.Tensor):
        raise TypeError("counts and mean must be torch tensors")
    if counts.ndim != 2 or mean.shape != counts.shape:
        raise ValueError("counts and mean must be aligned matrices")
    if counts.device != mean.device or counts.dtype != mean.dtype:
        raise ValueError("counts and mean must share device and dtype")
    if not torch.is_floating_point(counts):
        raise TypeError("counts and mean must use a floating dtype")
    if counts.dtype != torch.float64:
        raise TypeError("negative-binomial likelihood requires torch.float64")
    if not torch.isfinite(counts).all() or torch.any(counts < 0):
        raise ValueError("counts must be finite and nonnegative")
    if torch.any(counts > MAX_V28_COUNT_OR_LIBRARY):
        raise ValueError("counts exceed the stable v28 likelihood limit")
    if torch.any(counts != torch.floor(counts)):
        raise ValueError("counts must be integral")
    if not torch.isfinite(mean).all() or torch.any(mean < 0):
        raise ValueError("mean must be finite and nonnegative")
    if torch.any(mean > MAX_V28_COUNT_OR_LIBRARY):
        raise ValueError("mean exceeds the stable v28 likelihood limit")
    if not isinstance(inverse_dispersion, torch.Tensor):
        raise TypeError("inverse_dispersion must be a torch tensor")
    if (
        inverse_dispersion.device != counts.device
        or inverse_dispersion.dtype != counts.dtype
    ):
        raise ValueError("inverse_dispersion must share count device and dtype")
    if inverse_dispersion.ndim == 1:
        if inverse_dispersion.shape[0] != counts.shape[1]:
            raise ValueError("inverse_dispersion must contain one value per gene")
        size = inverse_dispersion.unsqueeze(0).expand_as(counts)
    elif inverse_dispersion.shape == counts.shape:
        size = inverse_dispersion
    else:
        raise ValueError("inverse_dispersion must be gene-wise or count-shaped")
    if not torch.isfinite(size).all() or torch.any(size <= 0):
        raise ValueError("inverse_dispersion must be finite and positive")
    if torch.any(size < MIN_V28_INVERSE_DISPERSION) or torch.any(
        size > MAX_V28_INVERSE_DISPERSION
    ):
        raise ValueError("inverse_dispersion exceeds stable v28 bounds")
    if mask is None:
        selected = torch.ones_like(counts, dtype=torch.bool)
    else:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch tensor")
        if mask.dtype != torch.bool or mask.shape != counts.shape:
            raise ValueError("mask must be boolean with the count shape")
        if mask.device != counts.device:
            raise ValueError("mask must share the count device")
        selected = mask
    if not torch.any(selected):
        raise ValueError("negative-binomial objective mask must not be empty")
    selected_counts = counts[selected]
    selected_mean = mean[selected]
    selected_size = size[selected]
    if torch.any((selected_counts > 0) & (selected_mean <= 0)):
        raise ValueError("positive selected counts require positive means")
    log_mean_input = torch.where(
        selected_counts > 0,
        selected_mean,
        torch.ones_like(selected_mean),
    )
    total = selected_size + selected_mean
    log_probability = (
        torch.lgamma(selected_counts + selected_size)
        - torch.lgamma(selected_size)
        - torch.lgamma(selected_counts + 1.0)
        + selected_size * (torch.log(selected_size) - torch.log(total))
        + selected_counts * torch.log(log_mean_input)
        - selected_counts * torch.log(total)
    )
    losses = -log_probability
    if not torch.isfinite(losses).all() or torch.any(losses < 0):
        raise FloatingPointError("negative-binomial likelihood produced invalid losses")
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


class NegativeBinomialMaskAutoencoder(nn.Module):
    """Explicit-mask encoder with a simplex decoder for NB count means."""

    def __init__(
        self,
        n_genes: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        if isinstance(n_genes, bool) or not isinstance(n_genes, int) or n_genes <= 0:
            raise ValueError("n_genes must be a positive integer")
        hidden = tuple(hidden_dims)
        if not hidden or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in hidden
        ):
            raise ValueError("hidden_dims must contain positive integers")
        if (
            isinstance(latent_dim, bool)
            or not isinstance(latent_dim, int)
            or latent_dim <= 0
        ):
            raise ValueError("latent_dim must be a positive integer")
        self.n_genes = n_genes
        self.mask_token = nn.Parameter(torch.zeros(n_genes))
        encoder_layers: list[nn.Module] = []
        previous = 2 * n_genes
        for width in hidden:
            encoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        encoder_layers.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers: list[nn.Module] = []
        previous = latent_dim
        for width in reversed(hidden):
            decoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        decoder_layers.append(nn.Linear(previous, n_genes))
        self.decoder = nn.Sequential(*decoder_layers)

    def prepare_encoder_input(
        self,
        normalized_expression: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_expression.ndim != 2:
            raise ValueError("normalized_expression must be two-dimensional")
        if normalized_expression.shape[1] != self.n_genes:
            raise ValueError("normalized_expression gene dimension differs")
        if availability.shape != normalized_expression.shape:
            raise ValueError("availability shape must match expression")
        if availability.dtype != torch.bool:
            raise TypeError("availability must be boolean")
        if availability.device != normalized_expression.device:
            raise ValueError("availability and expression must share a device")
        tokens = self.mask_token.to(dtype=normalized_expression.dtype).expand_as(
            normalized_expression
        )
        represented = torch.where(availability, normalized_expression, tokens)
        return torch.cat(
            (represented, availability.to(dtype=normalized_expression.dtype)),
            dim=1,
        )

    def forward(
        self,
        normalized_expression: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(
            self.prepare_encoder_input(normalized_expression, availability)
        )
        fractions = torch.softmax(self.decoder(latent), dim=1)
        return fractions, latent


def apply_library_size_offset(
    gene_fractions: torch.Tensor,
    library_sizes: torch.Tensor,
) -> torch.Tensor:
    """Convert decoded gene fractions to NB means on observed-count scale."""

    if not isinstance(gene_fractions, torch.Tensor) or gene_fractions.ndim != 2:
        raise TypeError("gene_fractions must be a two-dimensional torch tensor")
    if not isinstance(library_sizes, torch.Tensor) or library_sizes.ndim != 1:
        raise TypeError("library_sizes must be a one-dimensional torch tensor")
    if library_sizes.shape[0] != gene_fractions.shape[0]:
        raise ValueError("library_sizes must contain one value per cell")
    if (
        library_sizes.device != gene_fractions.device
        or library_sizes.dtype != gene_fractions.dtype
    ):
        raise ValueError("library_sizes must share fraction device and dtype")
    if (
        not torch.isfinite(gene_fractions).all()
        or torch.any(gene_fractions < 0)
        or not torch.isfinite(library_sizes).all()
        or torch.any(library_sizes < 0)
    ):
        raise ValueError("fractions and library sizes must be finite and nonnegative")
    sums = gene_fractions.sum(dim=1)
    if not torch.allclose(
        sums,
        torch.ones_like(sums),
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError("gene_fractions must sum to one per cell")
    return gene_fractions * library_sizes[:, None]


def _negative_binomial_objective(
    dispersion: GeneDispersionEstimate,
    *,
    normalization_target: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build the fixed training objective after validation-safe estimation."""

    if type(dispersion) is not GeneDispersionEstimate:
        raise TypeError("dispersion must be an exact GeneDispersionEstimate")
    target = _finite_real(
        normalization_target,
        "normalization_target",
        positive=True,
    )
    inverse_dispersion = torch.as_tensor(
        np.array(dispersion.inverse_dispersion, copy=True),
        dtype=dtype,
        device=device,
    )

    def objective(
        fractions: torch.Tensor,
        counts: torch.Tensor,
        library_sizes: torch.Tensor,
        artificial_mask: torch.Tensor,
        natural_zero_mask: torch.Tensor,
        p_pre_zero: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from maskimpute.train import natural_zero_preservation_loss

        likelihood_fractions = fractions.to(dtype=counts.dtype)
        means = apply_library_size_offset(likelihood_fractions, library_sizes)
        if torch.any(artificial_mask):
            primary = negative_binomial_nll(
                counts,
                means,
                inverse_dispersion,
                mask=artificial_mask,
                reduction="mean",
            )
        else:
            primary = means.sum() * 0.0
        normalized_prediction = torch.log1p(fractions * target)
        preservation = natural_zero_preservation_loss(
            normalized_prediction,
            natural_zero_mask,
            p_pre_zero,
        )
        return primary, preservation

    return objective


__all__ = [
    "GeneDispersionEstimate",
    "MAX_V28_COUNT_OR_LIBRARY",
    "NegativeBinomialDecoderConfig",
    "NegativeBinomialMaskAutoencoder",
    "apply_library_size_offset",
    "estimate_shrunk_gene_dispersion",
    "negative_binomial_nll",
]
