"""Digest-free method output contracts for direct fair-comparator execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .base import MethodContractError, MethodInput, MethodSpec, _matrix_values


@dataclass(frozen=True, slots=True)
class DirectMethodOutput:
    """One aligned native matrix without a content-identity summary."""

    method_id: str
    output_scale: str
    obs_ids: tuple[str, ...]
    var_ids: tuple[str, ...]
    shape: tuple[int, int]
    _matrix_bytes: bytes = field(repr=False)

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.frombuffer(self._matrix_bytes, dtype="<f8").reshape(self.shape)
        matrix.setflags(write=False)
        return matrix


@dataclass(frozen=True, slots=True)
class DirectAdapterExecution:
    """Direct adapter output and raw streams, with no legacy identity artifact."""

    output: DirectMethodOutput
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.output, DirectMethodOutput):
            raise TypeError("output must be a DirectMethodOutput")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise MethodContractError("direct adapter streams must be exact bytes")


def finalize_direct_method_output(
    spec: MethodSpec,
    method_input: MethodInput,
    matrix: object,
    *,
    output_scale: str,
    obs_ids: Sequence[str],
    var_ids: Sequence[str],
) -> DirectMethodOutput:
    """Validate and freeze one aligned output without deriving an identity."""

    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if output_scale != spec.output_scale:
        raise MethodContractError("output scale does not match method specification")
    output_obs_ids = tuple(obs_ids)
    output_var_ids = tuple(var_ids)
    if output_obs_ids != method_input.obs_ids:
        raise MethodContractError("output obs IDs do not match method input")
    if output_var_ids != method_input.var_ids:
        raise MethodContractError("output var IDs do not match method input")
    output = _matrix_values(matrix, "direct method output")
    if output.shape != method_input.shape:
        raise MethodContractError(
            f"method output shape {output.shape} does not match {method_input.shape}"
        )
    if not np.isfinite(output).all():
        raise MethodContractError("method output must contain finite values")
    if bool((output < 0).any()):
        raise MethodContractError("method output must be nonnegative")
    if spec.preserves_observed_positives:
        counts = method_input.counts
        positive = counts > 0
        if not np.array_equal(output[positive], counts[positive]):
            raise MethodContractError("method output must preserve observed positives")
    return DirectMethodOutput(
        method_id=spec.id,
        output_scale=output_scale,
        obs_ids=output_obs_ids,
        var_ids=output_var_ids,
        shape=method_input.shape,
        _matrix_bytes=np.asarray(output, dtype="<f8", order="C").tobytes(order="C"),
    )


__all__ = [
    "DirectAdapterExecution",
    "DirectMethodOutput",
    "finalize_direct_method_output",
]
