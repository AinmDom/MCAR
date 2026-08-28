"""Frozen MCAR v3.5.1 with a bounded FiLM-SIREN feature gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from mcar.models.film_siren import FilmSiren
from mcar.models.residual_mlp_cnn import ResidualMLPCNN


@dataclass(frozen=True)
class BoundedCorrectionConfig:
    """Configuration for the bounded output-space correction."""

    previous_weight: float = 0.3
    candidate_weight: float = 0.7
    maximum_mix: float = 0.5

    def validate(self) -> None:
        if self.previous_weight < 0.0 or self.candidate_weight < 0.0:
            raise ValueError("MCAR component weights must be nonnegative")
        if abs(self.previous_weight + self.candidate_weight - 1.0) > 1e-7:
            raise ValueError("MCAR component weights must sum to one")
        if not 0.0 < self.maximum_mix <= 1.0:
            raise ValueError("maximum_mix must be in (0, 1]")


class BoundedMcarFilmCorrection(nn.Module):
    """Apply a zero-initialized bounded FiLM correction to frozen MCAR.

    The trainable gate is evaluated from the final FiLM-SIREN features.  Its
    output is zero initialized, making the complete model exactly identical to
    the frozen MCAR v3.5.1 ensemble before optimization.  The correction is a
    bounded, query-dependent fraction of the frozen FiLM-minus-MCAR difference.
    """

    def __init__(
        self,
        film_siren: FilmSiren,
        previous_mcar: ResidualMLPCNN,
        candidate_mcar: ResidualMLPCNN,
        configuration: BoundedCorrectionConfig,
    ) -> None:
        super().__init__()
        configuration.validate()
        self.film_siren = film_siren
        self.previous_mcar = previous_mcar
        self.candidate_mcar = candidate_mcar
        self.correction_configuration = configuration
        self.gate_output = nn.Linear(
            film_siren.configuration.hidden_width,
            film_siren.configuration.output_dimension,
        )
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)
        self.freeze_backbones()

    @property
    def configuration(self):  # type: ignore[no-untyped-def]
        return self.film_siren.configuration

    @property
    def encoder_configuration(self):  # type: ignore[no-untyped-def]
        return self.film_siren.encoder_configuration

    def encode_condition(
        self,
        normalized_magnitude: torch.Tensor,
        xyz: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.film_siren.encode_condition(normalized_magnitude, xyz, mask)

    def modulation_statistics(self, latent: torch.Tensor) -> dict[str, object]:
        return self.film_siren.modulation_statistics(latent)

    def freeze_backbones(self) -> None:
        for backbone in (self.film_siren, self.previous_mcar, self.candidate_mcar):
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
            backbone.eval()

    def train(self, mode: bool = True) -> "BoundedMcarFilmCorrection":
        super().train(mode)
        self.film_siren.eval()
        self.previous_mcar.eval()
        self.candidate_mcar.eval()
        return self

    def forward_grid(
        self,
        coordinates: torch.Tensor,
        latent: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final, MCAR base, bounded correction, and gate tensors."""
        if point_features.ndim != 4 or point_features.shape[1] != 2:
            raise ValueError("point_features must have shape [D,2,F,7]")
        direction_count, _, frequency_count, _ = point_features.shape
        if coordinates.shape[0] != direction_count * frequency_count:
            raise ValueError("coordinate and MCAR grids do not match")

        with torch.no_grad():
            previous, _, _ = self.previous_mcar(point_features)
            candidate, _, _ = self.candidate_mcar(point_features)
            base = (
                self.correction_configuration.previous_weight * previous
                + self.correction_configuration.candidate_weight * candidate
            ).permute(1, 0, 2)
            film_prediction = self.film_siren(coordinates, latent).reshape(
                direction_count, frequency_count, 2
            ).permute(2, 0, 1)
            hidden = self.film_siren.forward_features(coordinates, latent)

        raw_gate = self.gate_output(hidden).reshape(
            direction_count, frequency_count, 2
        ).permute(2, 0, 1)
        gate = self.correction_configuration.maximum_mix * torch.tanh(raw_gate)
        correction = gate * (film_prediction - base)
        return base + correction, base, correction, gate

    def correction_configuration_dict(self) -> dict[str, float]:
        return asdict(self.correction_configuration)
