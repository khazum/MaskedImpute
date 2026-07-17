"""Prespecified, capacity-matched MaskImpute ablation contracts.

The ablation registry records component-level interventions before development
results exist.  It deliberately contains no efficacy weights or selection
score.  Helpers in this module make the declared masking, score, architecture,
gate, and output policies executable without exposing evaluator truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from maskimpute.config import MaskImputeConfig
from maskimpute.train import (
    _train_with_policies,
    _numeric_matrix_to_dense,
    invert_observed_normalization,
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
)
_TRACKED_ABLATION_REGISTRY = (
    Path(__file__).resolve().parents[1] / "study" / "ablations.json"
)
_TRACKED_ABLATION_REGISTRY_SHA256 = (
    "dd4da34e0ebe5e7eb349fac3ed89063781bcddf640b01601b9a3c82a2e43b26f"
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
    }
)


def _exact_mapping(value: object, expected: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != expected:
        raise ValueError(f"{name} has missing or extra fields")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
                "score_source": "retained_calibrator",
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


@dataclass(frozen=True, slots=True)
class AblationRunResult:
    """Neutral primary-output wrapper for selective and nonselective controls."""

    output_policy: str
    _result: object

    def __post_init__(self) -> None:
        from maskimpute.result import ImputationResult

        if self.output_policy not in {"selective", "full_gated", "full_ungated"}:
            raise ValueError("output_policy is invalid")
        if type(self._result) is not ImputationResult:
            raise TypeError("_result must be an exact ImputationResult")

    @property
    def primary_counts(self):
        return self._result.selective_counts  # type: ignore[attr-defined]

    @property
    def selective_counts(self):
        if self.output_policy != "selective":
            raise AttributeError(
                "nonselective ablation output is available only as primary_counts"
            )
        return self.primary_counts

    @property
    def denoised_counts(self):
        return self._result.denoised_counts  # type: ignore[attr-defined]

    @property
    def p_pre_zero(self) -> np.ndarray:
        return self._result.p_pre_zero  # type: ignore[attr-defined]

    @property
    def latent(self) -> np.ndarray:
        return self._result.latent  # type: ignore[attr-defined]

    @property
    def diagnostics(self):
        return self._result.diagnostics  # type: ignore[attr-defined]


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


def _parse_ablation_registry_bytes(raw: bytes) -> AblationRegistry:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {value}")
            ),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
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


def load_ablation_registry(path: str | Path) -> AblationRegistry:
    """Load the exact tracked ablation schema and reject silent denominator edits."""

    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ValueError("ablation registry is not readable canonical JSON") from error
    return _parse_ablation_registry_bytes(raw)


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
                # A nonlinear expression-only channel replaces the token parameters.
                # It retains active capacity without encoding availability itself.
                self.expression_curvature = nn.Parameter(torch.zeros(n_genes))
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
            curvature = self.expression_curvature.to(dtype=expression.dtype)
            expression_only = represented * torch.sigmoid(curvature * represented)
            return torch.cat((represented, expression_only), dim=1)

        def forward(self, expression, availability):
            latent = self.encoder(self.prepare_encoder_input(expression, availability))
            linear = self.decoder(latent)
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
    if spec.gate == "power_complement" and base.gate_gamma <= 0:
        raise ValueError("enabled power-complement gate requires positive gate_gamma")
    if spec.pre_zero_regularizer and base.pre_zero_regularization <= 0:
        raise ValueError(
            "enabled pre-zero regularizer requires positive pre_zero_regularization"
        )
    if spec.id == "no-gate" and base.gate_gamma <= 0:
        raise ValueError(
            "reference gate_gamma must be positive for the no-gate ablation"
        )
    if spec.id == "no-pre-zero-regularizer" and base.pre_zero_regularization <= 0:
        raise ValueError(
            "reference pre_zero_regularization must be positive for the "
            "no-pre-zero-regularizer ablation"
        )
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


def _training_config_payload(config: MaskImputeConfig) -> dict[str, object]:
    return {
        "artificial_mask_fraction": config.artificial_mask_fraction,
        "batch_size": config.batch_size,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "gate_gamma": config.gate_gamma,
        "hidden_dims": list(config.hidden_dims),
        "latent_dim": config.latent_dim,
        "learning_rate": config.learning_rate,
        "log_count_bin_edges": list(config.log_count_bin_edges),
        "max_epochs": config.max_epochs,
        "normalization_target": config.normalization_target,
        "patience": config.patience,
        "pre_zero_regularization": config.pre_zero_regularization,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "weight_decay": config.weight_decay,
    }


def _canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def resolve_score(
    p_pre_zero: object,
    observed_counts: object,
    spec: AblationSpec,
    *,
    calibrator: ScoreCalibrator | None,
) -> np.ndarray:
    """Resolve direct versus retained-calibrator score without reconstruction input."""

    counts = validate_observed_counts(observed_counts)
    probability = validate_p_pre_zero(p_pre_zero, counts)
    if spec.score_source == "direct":
        return probability
    from maskimpute.calibration import ScoreCalibrator

    if type(calibrator) is not ScoreCalibrator:
        raise ValueError("retained score variant requires an exact calibrator")
    calibrated = np.zeros_like(probability)
    observed_zero = counts == 0
    calibrated[observed_zero] = calibrator.transform(probability[observed_zero])
    return calibrated


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


def _trusted_ablation_spec(spec: object) -> tuple[AblationSpec, AblationRegistry, str]:
    if type(spec) is not AblationSpec:
        raise TypeError("spec must be an exact AblationSpec")
    try:
        registry_bytes = _TRACKED_ABLATION_REGISTRY.read_bytes()
    except OSError as error:
        raise RuntimeError(
            "tracked publication ablation registry is unavailable"
        ) from error
    registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    if registry_sha256 != _TRACKED_ABLATION_REGISTRY_SHA256:
        raise ValueError("tracked publication ablation registry digest differs")
    registry = _parse_ablation_registry_bytes(registry_bytes)
    trusted_by_id = {registry.reference.id: registry.reference, **registry.by_id}
    trusted = trusted_by_id.get(spec.id)
    if trusted is None or trusted != spec:
        raise ValueError("spec does not exactly match the tracked ablation registry")
    return trusted, registry, registry_sha256


def _derive_prezero_execution_policy(
    observed_counts: object,
    cell_ids: object,
    score_artifact: object,
    calibration_artifact: object,
    spec: AblationSpec,
    *,
    calibration_usage: str,
    development_mechanism: str,
    development_biological_id: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Derive the exact realized score matrix and its truth-free policy receipt."""

    from maskimpute.calibration import CalibrationArtifact, _canonical_json_bytes
    from maskimpute.count_model import PreZeroCountModelScore

    trusted_spec, _registry, _registry_sha256 = _trusted_ablation_spec(spec)
    mechanisms = {"symsim", "sergio", "sparsim", "semisynthetic"}
    if calibration_usage not in {
        "development_holdout",
        "retained_all_development",
    }:
        raise ValueError("calibration_usage is invalid")
    if calibration_usage == "development_holdout":
        if development_mechanism not in mechanisms:
            raise ValueError("development_mechanism is outside the tracked panel")
        if development_biological_id not in {"draw-01", "draw-02"}:
            raise ValueError("development_biological_id is outside the tracked panel")
    else:
        final_identity = (
            development_mechanism in mechanisms
            and development_biological_id
            in {f"draw-{draw:02d}" for draw in range(1, 6)}
        )
        external_identity = (
            development_mechanism
            in {
                "cite-seq-cbmc-rna-protein",
                "tung-ipsc-ercc-bulk-replicates",
            }
            and development_biological_id == "external"
        )
        if not (final_identity or external_identity):
            raise ValueError(
                "retained-all-development inference identity is outside authority"
            )

    counts = validate_observed_counts(observed_counts)
    if type(score_artifact) is not PreZeroCountModelScore:
        raise TypeError("score_artifact must be an exact PreZeroCountModelScore")
    direct_score = validate_p_pre_zero(
        score_artifact.score_for_counts(counts, cell_ids),
        counts,
    )
    if type(calibration_artifact) is not CalibrationArtifact:
        raise TypeError("calibration_artifact must be an exact CalibrationArtifact")
    verified_calibration = CalibrationArtifact(calibration_artifact.to_dict())
    calibration_payload = verified_calibration.to_dict()

    probability = np.array(direct_score, copy=True)
    calibration_scope = "not_applicable_direct_score"
    calibration_holdout = None
    calibration_fold_receipt = None
    if trusted_spec.score_source == "retained_calibrator":
        observed_zero = counts == 0
        if calibration_usage == "retained_all_development":
            probability[observed_zero] = verified_calibration.transform(
                direct_score[observed_zero]
            )
            calibration_scope = (
                "retained_all_development_for_external_inference"
                if development_biological_id == "external"
                else "retained_all_development_for_final_inference"
            )
        elif development_mechanism == "symsim":
            probability[observed_zero] = (
                verified_calibration.transform_for_development_holdout(
                    direct_score[observed_zero],
                    mechanism=development_mechanism,
                    biological_id=development_biological_id,
                )
            )
            calibration_scope = "leave_one_biological_draw_out"
            calibration_holdout = {
                "mechanism": development_mechanism,
                "biological_id": development_biological_id,
            }
            fold = next(
                value
                for value in calibration_payload["development_holdout_calibrators"]
                if value["mechanism"] == development_mechanism
                and value["biological_id"] == development_biological_id
            )
            calibration_fold_receipt = {
                "calibrator_algorithm": fold["calibrator"]["algorithm"],
                "calibrator_sha256": _canonical_payload_sha256(fold["calibrator"]),
                "held_out_manifest_sha256s": tuple(
                    fold["held_out_manifest_sha256s"]
                ),
                "training_manifest_sha256s": tuple(
                    fold["training_manifest_sha256s"]
                ),
            }
        else:
            probability[observed_zero] = verified_calibration.transform(
                direct_score[observed_zero]
            )
            calibration_scope = "all_development_external_exact_truth_mechanism"
        probability[counts > 0] = 0.0
        probability = validate_p_pre_zero(probability, counts)
        if verified_calibration.selected_algorithm == "identity":
            equivalence_reason = "retained_identity_calibrator_equals_direct_score"
        elif np.array_equal(probability, direct_score):
            equivalence_reason = (
                "retained_nonidentity_calibrator_unchanged_on_this_dataset"
            )
        else:
            equivalence_reason = "retained_nonidentity_calibrator_transformed_score"
    else:
        equivalence_reason = "direct_cross_fitted_count_score"

    score_manifest = score_artifact.manifest
    return probability, {
        "source": trusted_spec.score_source,
        "artifact_integrity_verified": True,
        "source_authorized_by_panel": False,
        "score_artifact_sha256": score_manifest["score_sha256"],
        "score_input_sha256": score_manifest["input_sha256"],
        "score_config_sha256": score_manifest["config_sha256"],
        "calibration_file_sha256": hashlib.sha256(
            _canonical_json_bytes(calibration_payload)
        ).hexdigest(),
        "calibration_payload_sha256": calibration_payload["payload_sha256"],
        "retained_calibrator": verified_calibration.selected_algorithm,
        "calibration_scope": calibration_scope,
        "calibration_holdout": calibration_holdout,
        "calibration_fold_receipt": calibration_fold_receipt,
        "equivalence_reason": equivalence_reason,
    }


def _fit_ablation_once(
    observed_counts: object,
    score_artifact: object,
    calibration_artifact: object,
    spec: AblationSpec,
    config: MaskImputeConfig,
    device: object,
    *,
    cell_ids: object,
    development_mechanism: str,
    development_biological_id: str,
    calibration_usage: str = "development_holdout",
    decoder: str = "scaled_gaussian",
    decoder_config: object | None = None,
    structure_config: object | None = None,
):
    """Fit one development ablation from verified, truth-free score artifacts.

    This internal single-run primitive does not authorize a publication panel.
    The authority layer must bind the dataset, common base configuration, count
    score, calibration artifact, complete spec-by-seed grid, and output manifest.
    This function still rejects raw score matrices and free-form interventions.
    """

    import torch

    from maskimpute.result import ImputationResult
    from maskimpute.train import TrainingOutcome

    trusted_spec, registry, registry_sha256 = _trusted_ablation_spec(spec)
    if decoder not in {"scaled_gaussian", "negative_binomial"}:
        raise ValueError("decoder must be scaled_gaussian or negative_binomial")
    if decoder == "scaled_gaussian" and (
        decoder_config is not None or structure_config is not None
    ):
        raise ValueError("scaled_gaussian does not accept decoder/structure config")
    if decoder == "negative_binomial" and trusted_spec.id != registry.reference.id:
        raise ValueError("negative_binomial is a reference-only development revision")
    if type(config) is not MaskImputeConfig:
        raise TypeError("config must be an exact MaskImputeConfig")
    config = replace(config)
    if config.seed not in registry.model_seeds:
        raise ValueError("config seed is outside the tracked ablation seed panel")
    resolved_config = resolve_training_config(config, trusted_spec)

    counts = validate_observed_counts(observed_counts)
    probability, score_diagnostics = _derive_prezero_execution_policy(
        counts,
        cell_ids,
        score_artifact,
        calibration_artifact,
        trusted_spec,
        calibration_usage=calibration_usage,
        development_mechanism=development_mechanism,
        development_biological_id=development_biological_id,
    )

    dispersion = None
    if decoder == "scaled_gaussian":

        def model_factory(
            n_genes: int,
            training_config: MaskImputeConfig,
        ) -> torch.nn.Module:
            return build_capacity_matched_model(
                n_genes=n_genes,
                hidden_dims=training_config.hidden_dims,
                latent_dim=training_config.latent_dim,
                encoder_mode=trusted_spec.encoder_mode,
            )

        if trusted_spec.positive_masking == "uniform":

            def training_mask_factory(
                values: object,
                *,
                validation_mask: object,
                fraction: float,
                log_count_bin_edges: Sequence[float],
                rng: np.random.Generator,
            ) -> np.ndarray:
                del log_count_bin_edges
                return make_uniform_positive_mask(
                    values,
                    validation_mask=validation_mask,
                    fraction=fraction,
                    rng=rng,
                )

        else:
            from maskimpute.train import make_epoch_training_mask

            training_mask_factory = make_epoch_training_mask

        outcome: TrainingOutcome = _train_with_policies(
            counts,
            probability,
            resolved_config,
            device,
            model_factory=model_factory,
            training_mask_factory=training_mask_factory,
        )
    else:
        from maskimpute.nb_model import NegativeBinomialDecoderConfig
        from maskimpute.train import train_v28, train_v29

        if type(decoder_config) is not NegativeBinomialDecoderConfig:
            raise TypeError(
                "negative_binomial decoder_config must be an exact "
                "NegativeBinomialDecoderConfig"
            )
        if structure_config is None:
            revised_outcome = train_v28(
                counts,
                probability,
                resolved_config,
                device,
                decoder_config=decoder_config,
            )
            structure = None
        else:
            from maskimpute.structure import StructurePenaltyConfig

            if type(structure_config) is not StructurePenaltyConfig:
                raise TypeError(
                    "structure_config must be an exact StructurePenaltyConfig"
                )
            revised_outcome = train_v29(
                counts,
                probability,
                resolved_config,
                device,
                decoder_config=decoder_config,
                structure_config=structure_config,
            )
            structure = revised_outcome.structure
        outcome = revised_outcome.training
        dispersion = revised_outcome.dispersion

    selected_device = next(outcome.model.parameters()).device
    expression = torch.as_tensor(
        outcome.normalized_expression,
        dtype=torch.float32,
        device=selected_device,
    )
    availability = torch.as_tensor(
        counts > 0,
        dtype=torch.bool,
        device=selected_device,
    )
    outcome.model.eval()
    with torch.no_grad():
        decoder_prediction, latent = outcome.model(expression, availability)
    prediction_dense = decoder_prediction.detach().cpu().numpy().astype(np.float64)
    latent_dense = latent.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(prediction_dense)) or np.any(prediction_dense < 0):
        raise FloatingPointError("ablation decoder produced invalid predictions")
    if not np.all(np.isfinite(latent_dense)):
        raise FloatingPointError("ablation encoder produced invalid latent values")
    if decoder == "scaled_gaussian":
        with np.errstate(over="ignore", invalid="ignore"):
            denoised_counts = invert_observed_normalization(
                prediction_dense,
                outcome.library_sizes,
                target=resolved_config.normalization_target,
            )
        invalid_decoder_message = (
            "ablation inverse normalization produced invalid counts"
        )
        method_version = "v27-development-ablation-single-run"
        primary_loss_name = "artificially_masked_observed_positive_mse"
        decoder_diagnostics: dict[str, object] = {
            "family": "scaled_gaussian",
            "prediction_scale": "log_normalized_expression",
            "count_conversion": "inverse_observed_library_log_normalization",
        }
    else:
        from maskimpute.nb_model import apply_library_size_offset

        libraries = torch.as_tensor(
            outcome.library_sizes,
            dtype=torch.float64,
            device=selected_device,
        )
        denoised_counts = (
            apply_library_size_offset(decoder_prediction.to(torch.float64), libraries)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        invalid_decoder_message = "negative-binomial decoder produced invalid counts"
        method_version = (
            "v28-development-candidate-single-run"
            if structure_config is None
            else "v29-development-candidate-single-run"
        )
        primary_loss_name = (
            "artificially_masked_observed_positive_negative_binomial_nll"
        )
        assert dispersion is not None
        dispersion_bytes = np.asarray(
            dispersion.dispersion,
            dtype="<f8",
        ).tobytes(order="C")
        decoder_diagnostics = {
            "family": "negative_binomial",
            "parameterization": "variance_equals_mean_plus_mean_squared_over_theta",
            "mean": "observed_library_size_times_decoded_gene_fraction",
            "gene_fraction_link": "softmax",
            "dispersion_estimator": (
                "exposure_adjusted_winsorized_moments_log_shrunk_to_gene_median"
            ),
            "dispersion_config": decoder_config.to_dict(),
            "global_dispersion": dispersion.global_dispersion,
            "gene_dispersion_min": float(np.min(dispersion.dispersion)),
            "gene_dispersion_max": float(np.max(dispersion.dispersion)),
            "gene_dispersion_sha256": hashlib.sha256(dispersion_bytes).hexdigest(),
            "validation_positive_values_excluded_from_dispersion": True,
        }
        if structure_config is not None:
            assert structure is not None
            decoder_diagnostics["structure_preservation"] = {
                "variable_gene_panel": "observed_count_variance_fixed_before_training",
                "neighborhood": "observed_input_knn_fixed_before_training",
                "variable_gene_sha256": structure.variable_gene_sha256,
                "neighborhood_sha256": structure.neighborhood_sha256,
                "config": structure_config.to_dict(),
                "truth_or_evaluation_labels_used": False,
                "additional_trainable_parameters": 0,
            }
    if not np.all(np.isfinite(denoised_counts)) or np.any(denoised_counts < 0):
        raise FloatingPointError(invalid_decoder_message)
    output_counts = apply_ablation_output(
        denoised_counts,
        counts,
        probability,
        trusted_spec,
        gamma=resolved_config.gate_gamma,
    )

    base_config_payload = _training_config_payload(config)
    effective_config_payload = _training_config_payload(resolved_config)
    diagnostics = {
        "method_version": method_version,
        "decoder": decoder_diagnostics,
        "ablation": {
            "id": trusted_spec.id,
            "changed_component": trusted_spec.changed_component,
            "encoder_mode": trusted_spec.encoder_mode,
            "gate": trusted_spec.gate,
            "output_policy": trusted_spec.output_policy,
            "pre_zero_regularizer": trusted_spec.pre_zero_regularizer,
            "encoder_interpretation": (
                "broader_expression_only_encoder_representation_with_active_"
                "capacity_compensation"
                if trusted_spec.encoder_mode == "implicit_numeric_zero"
                else "explicit_availability_indicator_and_learned_mask_token"
            ),
            "nominal_parameter_count": sum(
                parameter.numel() for parameter in outcome.model.parameters()
            ),
            "registry_sha256": registry_sha256,
        },
        "score": score_diagnostics,
        "masks": {
            "fixed_validation_mask_sha256": outcome.validation_mask_hashes[0],
            "fixed_validation_positive_entries": int(
                np.count_nonzero(outcome.validation_mask)
            ),
            "epoch_positive_masking": trusted_spec.positive_masking,
            "epoch_training_mask_sha256": outcome.epoch_training_mask_hashes,
        },
        "losses": {
            "primary": primary_loss_name,
            "natural_zero_penalty": (
                "mean(p_pre_zero * normalized_prediction_squared)"
            ),
            "natural_zero_penalty_weight": resolved_config.pre_zero_regularization,
            "training": outcome.training_loss_history,
            "validation": outcome.validation_loss_history,
        },
        "budget": {
            "optimization_signature": optimization_budget_signature(resolved_config),
            "base_config": base_config_payload,
            "base_config_sha256": _canonical_payload_sha256(base_config_payload),
            "effective_config": effective_config_payload,
            "effective_config_sha256": _canonical_payload_sha256(
                effective_config_payload
            ),
            "best_epoch": outcome.best_epoch,
            "stopped_epoch": outcome.stopped_epoch,
            "model_seed": resolved_config.seed,
        },
        "gate": {
            "family": trusted_spec.gate,
            "formula": (
                "prediction * (1 - p_pre_zero) ** gamma"
                if trusted_spec.gate == "power_complement"
                else "identity"
            ),
            "gamma": resolved_config.gate_gamma,
        },
        "primary_output_policy": trusted_spec.output_policy,
        "device": outcome.device,
    }
    return AblationRunResult(
        output_policy=trusted_spec.output_policy,
        _result=ImputationResult(
            selective_counts=output_counts,
            denoised_counts=denoised_counts,
            p_pre_zero=probability,
            latent=latent_dense,
            diagnostics=diagnostics,
        ),
    )
