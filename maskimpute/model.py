"""Explicit-mask autoencoder used by MaskImpute v27."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ExplicitMaskAutoencoder(nn.Module):
    """Autoencoder that represents unavailable genes with learned tokens.

    The encoder receives both the token-substituted expression matrix and an
    explicit availability channel.  Consequently, the numeric payload stored
    at unavailable positions cannot influence either the latent state or the
    reconstruction.
    """

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
        decoder_layers.extend((nn.Linear(previous, n_genes), nn.Softplus()))
        self.decoder = nn.Sequential(*decoder_layers)

    def prepare_encoder_input(
        self,
        normalized_expression: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """Build the token-substituted expression and indicator channels."""

        if normalized_expression.ndim != 2:
            raise ValueError("normalized_expression must be a two-dimensional tensor")
        if normalized_expression.shape[1] != self.n_genes:
            raise ValueError(
                "normalized_expression gene dimension does not match model"
            )
        if availability.shape != normalized_expression.shape:
            raise ValueError("availability shape must match normalized_expression")
        if availability.dtype != torch.bool:
            raise TypeError("availability must be a boolean tensor")
        if availability.device != normalized_expression.device:
            raise ValueError(
                "availability and normalized_expression must share a device"
            )

        tokens = self.mask_token.to(dtype=normalized_expression.dtype).expand_as(
            normalized_expression
        )
        represented = torch.where(availability, normalized_expression, tokens)
        indicator = availability.to(dtype=normalized_expression.dtype)
        return torch.cat((represented, indicator), dim=1)

    def forward(
        self,
        normalized_expression: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a nonnegative normalized reconstruction and latent state."""

        encoder_input = self.prepare_encoder_input(
            normalized_expression,
            availability,
        )
        latent = self.encoder(encoder_input)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
