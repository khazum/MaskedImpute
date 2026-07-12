"""Prespecified, capacity-matched MaskImpute ablation contracts.

The ablation registry records component-level interventions before development
results exist.  It deliberately contains no efficacy weights or selection
score.  Helpers in this module make the declared masking, score, architecture,
gate, and output policies executable without exposing evaluator truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from maskimpute.config import MaskImputeConfig
from maskimpute.train import (
    _numeric_matrix_to_dense,
    validate_observed_counts,
    validate_p_pre_zero,
)

if TYPE_CHECKING:
    from maskimpute.calibration import ScoreCalibrator


_VARIANT_ORDER = (
    "capacity-matched-ae",
    "no-gate",
    "no-pre-zero-regularizer",
    "no-explicit-mask",
    "full-denoising",
    "direct-score",
    "calibrated-score",
)
_SPEC_FIELDS = {
    "id",
    "changed_component",
    "positive_masking",
    "pre_zero_regularizer",
    "encoder_mode",
    "gate",
    "output_policy",
    "score_source",
}
_COMPONENT_FIELDS = (
    "positive_masking",
    "pre_zero_regularizer",
    "encoder_mode",
    "gate",
    "output_policy",
    "score_source",
)
_EXPECTED_SINGLE_CHANGES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "no-gate": frozenset({"gate"}),
        "no-pre-zero-regularizer": frozenset({"pre_zero_regularizer"}),
        "no-explicit-mask": frozenset({"encoder_mode"}),
        "full-denoising": frozenset({"output_policy"}),
        "direct-score": frozenset({"score_source"}),
        "calibrated-score": frozenset(),
    }
)


def _exact_mapping(value: object, expected: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != expected:
        raise ValueError(f"{name} has missing or extra fields")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One named intervention relative to the frozen MaskImpute candidate."""

    id: str
    changed_component: str
    positive_masking: str
    pre_zero_regularizer: bool
    encoder_mode: str
    gate: str
    output_policy: str
    score_source: str

    def __post_init__(self) -> None:
        for name in ("id", "changed_component"):
            object.__setattr__(
                self,
                name,
                _nonempty_string(getattr(self, name), name),
            )
        if self.positive_masking not in {"uniform", "log_count_stratified"}:
            raise ValueError("positive_masking is invalid")
        if type(self.pre_zero_regularizer) is not bool:
            raise ValueError("pre_zero_regularizer must be boolean")
        if self.encoder_mode not in {"explicit_mask", "implicit_numeric_zero"}:
            raise ValueError("encoder_mode is invalid")
        if self.gate not in {"none", "power_complement"}:
            raise ValueError("gate is invalid")
        if self.output_policy not in {
            "selective",
            "full_gated",
            "full_ungated",
        }:
            raise ValueError("output_policy is invalid")
        if self.score_source not in {"direct", "retained_calibrator"}:
            raise ValueError("score_source is invalid")
        if self.output_policy == "full_ungated" and self.gate != "none":
            raise ValueError("full_ungated output requires gate=none")
        if self.output_policy == "full_gated" and self.gate == "none":
            raise ValueError("full_gated output requires a gate")

    def validate_against_reference(self, reference: AblationSpec) -> None:
        """Reject undeclared multi-component changes."""

        if not isinstance(reference, AblationSpec):
            raise TypeError("reference must be an AblationSpec")
        if self.id == reference.id:
            if self != reference:
                raise ValueError("reference specification cannot differ from reference")
            return
        if self.id == "capacity-matched-ae":
            expected = {
                "positive_masking": "uniform",
                "pre_zero_regularizer": False,
                "encoder_mode": "explicit_mask",
                "gate": "none",
                "output_policy": "full_ungated",
                "score_source": "direct",
            }
            if any(getattr(self, field) != value for field, value in expected.items()):
                raise ValueError("capacity-matched control bundle is not prespecified")
            return
        if self.id not in _EXPECTED_SINGLE_CHANGES:
            raise ValueError("variant is not in the prespecified ablation panel")
        changed = frozenset(
            field
            for field in _COMPONENT_FIELDS
            if getattr(self, field) != getattr(reference, field)
        )
        if changed != _EXPECTED_SINGLE_CHANGES[self.id]:
            raise ValueError(
                f"{self.id} changes {sorted(changed)}, not its declared component"
            )


@dataclass(frozen=True, slots=True)
class AblationRegistry:
    """Immutable publication ablation denominator and budget declaration."""

    schema_version: int
    model_seeds: tuple[int, ...]
    parameter_budget: str
    optimizer_budget: str
    preprocessing_budget: str
    reference: AblationSpec
    variants: tuple[AblationSpec, ...]

    @property
    def by_id(self) -> dict[str, AblationSpec]:
        return {spec.id: spec for spec in self.variants}


def _parse_spec(value: object, name: str) -> AblationSpec:
    payload = _exact_mapping(value, _SPEC_FIELDS, name)
    return AblationSpec(
        id=payload["id"],  # type: ignore[arg-type]
        changed_component=payload["changed_component"],  # type: ignore[arg-type]
        positive_masking=payload["positive_masking"],  # type: ignore[arg-type]
        pre_zero_regularizer=payload["pre_zero_regularizer"],  # type: ignore[arg-type]
        encoder_mode=payload["encoder_mode"],  # type: ignore[arg-type]
        gate=payload["gate"],  # type: ignore[arg-type]
        output_policy=payload["output_policy"],  # type: ignore[arg-type]
        score_source=payload["score_source"],  # type: ignore[arg-type]
    )


def load_ablation_registry(path: str | Path) -> AblationRegistry:
    """Load the exact tracked ablation schema and reject silent denominator edits."""

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("ablation registry is not readable canonical JSON") from error
    root = _exact_mapping(
        payload,
        {
            "schema_version",
            "model_seeds",
            "parameter_budget",
            "optimizer_budget",
            "preprocessing_budget",
            "reference",
            "variants",
        },
        "ablation registry",
    )
    if root["schema_version"] != 1 or type(root["schema_version"]) is not int:
        raise ValueError("schema_version must equal 1")
    seed_values = root["model_seeds"]
    if type(seed_values) is not list or any(
        type(seed) is not int for seed in seed_values
    ):
        raise ValueError("model_seeds must be a JSON list of integers")
    seeds = tuple(seed_values)
    if seeds != (42, 43, 44) or len(seeds) != len(set(seeds)):
        raise ValueError("model_seeds must equal the prespecified unique seeds")
    budget_values = {
        "parameter_budget": "exact_nominal_match",
        "optimizer_budget": "shared_frozen_candidate_budget",
        "preprocessing_budget": "shared_except_named_component",
    }
    for field, expected in budget_values.items():
        if root[field] != expected:
            raise ValueError(f"{field} differs from the prespecified budget")
    reference = _parse_spec(root["reference"], "reference")
    if (
        reference.id != "maskimpute-reference"
        or reference.changed_component != "reference"
    ):
        raise ValueError("reference identifier and role must be prespecified")
    variant_values = root["variants"]
    if type(variant_values) is not list:
        raise ValueError("variants must be a JSON list")
    variants = tuple(
        _parse_spec(value, f"variants[{index}]")
        for index, value in enumerate(variant_values)
    )
    identifiers = tuple(spec.id for spec in variants)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("variants contain a duplicate identifier")
    if identifiers != _VARIANT_ORDER:
        raise ValueError("variants differ from the prespecified ordered denominator")
    expected_components = {
        "capacity-matched-ae": "control_bundle",
        "no-gate": "gate",
        "no-pre-zero-regularizer": "pre_zero_regularizer",
        "no-explicit-mask": "encoder_mode",
        "full-denoising": "output_policy",
        "direct-score": "score_source",
        "calibrated-score": "score_source_control",
    }
    for spec in variants:
        if spec.changed_component != expected_components[spec.id]:
            raise ValueError(f"{spec.id} changed_component is not prespecified")
        spec.validate_against_reference(reference)
    return AblationRegistry(
        schema_version=1,
        model_seeds=seeds,
        parameter_budget=root["parameter_budget"],  # type: ignore[arg-type]
        optimizer_budget=root["optimizer_budget"],  # type: ignore[arg-type]
        preprocessing_budget=root["preprocessing_budget"],  # type: ignore[arg-type]
        reference=reference,
        variants=variants,
    )


def build_capacity_matched_model(
    *,
    n_genes: int,
    hidden_dims: Sequence[int],
    latent_dim: int,
    encoder_mode: str,
):
    """Build an executable explicit/implicit pair with equal parameter count."""

    import torch
    from torch import nn

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
    if encoder_mode not in {"explicit_mask", "implicit_numeric_zero"}:
        raise ValueError("encoder_mode is invalid")

    class CapacityMatchedAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_genes = n_genes
            self.encoder_mode = encoder_mode
            if encoder_mode == "explicit_mask":
                self.mask_token = nn.Parameter(torch.zeros(n_genes))
            else:
                # This active output offset replaces the explicit model's token
                # parameters, retaining nominal capacity without encoding missingness.
                self.output_offset = nn.Parameter(torch.zeros(n_genes))
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

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

        def prepare_encoder_input(self, expression, availability):
            if expression.ndim != 2 or expression.shape[1] != self.n_genes:
                raise ValueError("expression must have the configured gene dimension")
            if (
                availability.shape != expression.shape
                or availability.dtype != torch.bool
            ):
                raise ValueError(
                    "availability must be boolean with the expression shape"
                )
            if availability.device != expression.device:
                raise ValueError("availability and expression must share a device")
            if self.encoder_mode == "explicit_mask":
                token = self.mask_token.to(dtype=expression.dtype).expand_as(expression)
                represented = torch.where(availability, expression, token)
                return torch.cat(
                    (represented, availability.to(expression.dtype)), dim=1
                )
            represented = torch.where(
                availability, expression, torch.zeros_like(expression)
            )
            return torch.cat((represented, represented), dim=1)

        def forward(self, expression, availability):
            latent = self.encoder(self.prepare_encoder_input(expression, availability))
            linear = self.decoder(latent)
            if self.encoder_mode == "implicit_numeric_zero":
                linear = linear + self.output_offset.to(dtype=linear.dtype)
            return torch.nn.functional.softplus(linear), latent

    return CapacityMatchedAutoencoder()


def make_uniform_positive_mask(
    observed_counts: object,
    *,
    validation_mask: object,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly mask training positives for the capacity-matched control."""

    counts = validate_observed_counts(observed_counts)
    if np.ma.isMaskedArray(validation_mask):
        raise TypeError("validation_mask must not be a masked array")
    validation = np.asarray(validation_mask)
    if validation.dtype != np.bool_ or validation.shape != counts.shape:
        raise ValueError("validation_mask must be boolean with the count-matrix shape")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float, np.number)):
        raise TypeError("fraction must be a real number")
    fraction_value = float(fraction)
    if not math.isfinite(fraction_value) or not 0 < fraction_value < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    candidates = np.flatnonzero(((counts > 0) & ~validation).ravel())
    if not candidates.size:
        raise ValueError("no training positives remain outside validation")
    number = max(
        1,
        min(candidates.size, int(np.floor(candidates.size * fraction_value + 0.5))),
    )
    chosen = rng.choice(candidates, size=number, replace=False)
    result = np.zeros(counts.size, dtype=np.bool_)
    result[chosen] = True
    return result.reshape(counts.shape)


def resolve_training_config(
    base: MaskImputeConfig,
    spec: AblationSpec,
) -> MaskImputeConfig:
    """Change only the named loss term while retaining the frozen budget."""

    if type(base) is not MaskImputeConfig:
        raise TypeError("base must be an exact MaskImputeConfig")
    if type(spec) is not AblationSpec:
        raise TypeError("spec must be an exact AblationSpec")
    regularization = base.pre_zero_regularization if spec.pre_zero_regularizer else 0.0
    return replace(base, pre_zero_regularization=regularization)


def optimization_budget_signature(config: MaskImputeConfig) -> tuple[object, ...]:
    """Return all architecture, optimizer, data, and stopping budget fields."""

    if type(config) is not MaskImputeConfig:
        raise TypeError("config must be an exact MaskImputeConfig")
    return (
        config.hidden_dims,
        config.latent_dim,
        config.learning_rate,
        config.weight_decay,
        config.batch_size,
        config.max_epochs,
        config.patience,
        config.artificial_mask_fraction,
        config.validation_fraction,
        config.log_count_bin_edges,
        config.early_stopping_min_delta,
        config.normalization_target,
        config.seed,
    )


def resolve_score(
    p_pre_zero: object,
    spec: AblationSpec,
    *,
    calibrator: ScoreCalibrator | None,
) -> np.ndarray:
    """Resolve direct versus retained-calibrator score without reconstruction input."""

    probability, _ = _numeric_matrix_to_dense(p_pre_zero, "p_pre_zero")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("p_pre_zero must lie in [0, 1]")
    if spec.score_source == "direct":
        return probability
    from maskimpute.calibration import ScoreCalibrator

    if type(calibrator) is not ScoreCalibrator:
        raise ValueError("retained score variant requires an exact calibrator")
    return calibrator.transform(probability)


def apply_ablation_output(
    candidates: object,
    observed_counts: object,
    p_pre_zero: object,
    spec: AblationSpec,
    *,
    gamma: float,
) -> np.ndarray:
    """Apply only the declared gate and positive-copy output intervention."""

    if type(spec) is not AblationSpec:
        raise TypeError("spec must be an exact AblationSpec")
    counts = validate_observed_counts(observed_counts)
    prediction, _ = _numeric_matrix_to_dense(candidates, "candidates")
    if prediction.shape != counts.shape:
        raise ValueError("candidate and observed shapes must match")
    if np.any(prediction < 0):
        raise ValueError("candidates must be nonnegative")
    if spec.output_policy == "full_ungated":
        return prediction
    if spec.gate == "power_complement":
        probability = validate_p_pre_zero(p_pre_zero, counts)
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float, np.number)):
            raise TypeError("gamma must be a real number")
        exponent = float(gamma)
        if not math.isfinite(exponent) or exponent < 0:
            raise ValueError("gamma must be finite and nonnegative")
        output = prediction * np.power(1.0 - probability, exponent)
    else:
        output = np.array(prediction, copy=True)
    if spec.output_policy == "selective":
        output[counts > 0] = counts[counts > 0]
    return output
