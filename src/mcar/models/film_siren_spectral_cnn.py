"""FiLM-SIREN base field with a zero-initialized binaural spectral refiner."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from mcar.models.film_siren import FilmSiren
from mcar.models.residual_mlp_cnn import BinauralSpectralCNN


@dataclass(frozen=True)
class SpectralRefinerConfig:
    """Architecture of the MCAR-compatible local spectral refiner."""

    input_channels: int = 7
    channels: int = 48
    dilation_schedule: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 7
    condition_width: int = 64

    def validate(self) -> None:
        if self.input_channels != 7:
            raise ValueError("The frozen MCAR spectral interface requires 7 channels")
        if min(self.channels, self.kernel_size, self.condition_width) < 1:
            raise ValueError("Spectral-refiner dimensions must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if not self.dilation_schedule or any(
            int(dilation) < 1 for dilation in self.dilation_schedule
        ):
            raise ValueError("dilation_schedule must contain positive integers")


class FilmSirenSpectralCNN(nn.Module):
    """Refine a frozen FiLM-SIREN residual with the MCAR spectral CNN.

    The seven spectral channels are, in order, normalized left MCA,
    normalized left correction, left FiLM-SIREN base residual, normalized
    right MCA, normalized right correction, right FiLM-SIREN base residual,
    and normalized log-frequency.  The CNN output layer is zero initialized by
    :class:`BinauralSpectralCNN`, so a newly constructed hybrid is exactly the
    supplied FiLM-SIREN at every query.
    """

    def __init__(
        self,
        film_siren: FilmSiren,
        refiner_configuration: SpectralRefinerConfig,
        *,
        freeze_film_siren: bool = True,
    ) -> None:
        super().__init__()
        refiner_configuration.validate()
        self.film_siren = film_siren
        self.refiner_configuration = refiner_configuration
        self.spectral_cnn = BinauralSpectralCNN(
            input_channels=refiner_configuration.input_channels,
            channels=refiner_configuration.channels,
            dilation_schedule=refiner_configuration.dilation_schedule,
            kernel_size=refiner_configuration.kernel_size,
            condition_width=refiner_configuration.condition_width,
        )
        self.film_siren_frozen = False
        if freeze_film_siren:
            self.freeze_film_siren()

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

    def freeze_film_siren(self) -> None:
        for parameter in self.film_siren.parameters():
            parameter.requires_grad_(False)
        self.film_siren.eval()
        self.film_siren_frozen = True

    def unfreeze_film_siren(self) -> None:
        for parameter in self.film_siren.parameters():
            parameter.requires_grad_(True)
        self.film_siren_frozen = False

    def train(self, mode: bool = True) -> "FilmSirenSpectralCNN":
        super().train(mode)
        if self.film_siren_frozen:
            self.film_siren.eval()
        return self

    def forward_grid(
        self,
        coordinates: torch.Tensor,
        latent: torch.Tensor,
        normalized_mca: torch.Tensor,
        normalized_correction: torch.Tensor,
        normalized_log_frequency: torch.Tensor,
        direction_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final, base, and delta tensors with shape ``[2,D,F]``."""
        if coordinates.ndim != 2:
            raise ValueError("coordinates must have shape [D*F,C]")
        if normalized_mca.ndim != 3 or normalized_mca.shape[0] != 2:
            raise ValueError("normalized_mca must have shape [2,D,F]")
        if normalized_correction.shape != normalized_mca.shape:
            raise ValueError("normalized_correction must match normalized_mca")
        _, direction_count, frequency_count = normalized_mca.shape
        if coordinates.shape[0] != direction_count * frequency_count:
            raise ValueError("coordinate and spectral grid sizes do not match")
        if normalized_log_frequency.shape != (frequency_count,):
            raise ValueError("normalized_log_frequency must have shape [F]")
        if direction_xyz.shape != (direction_count, 3):
            raise ValueError("direction_xyz must have shape [D,3]")

        base_flat = self.film_siren(coordinates, latent)
        if base_flat.shape != (direction_count * frequency_count, 2):
            raise ValueError("FiLM-SIREN output does not match the query grid")
        base = base_flat.reshape(direction_count, frequency_count, 2).permute(2, 0, 1)
        spectrum_features = torch.stack(
            (
                normalized_mca[0],
                normalized_correction[0],
                base[0],
                normalized_mca[1],
                normalized_correction[1],
                base[1],
                normalized_log_frequency[None, :].expand(direction_count, -1),
            ),
            dim=1,
        )
        delta = self.spectral_cnn(spectrum_features, direction_xyz).permute(1, 0, 2)
        return base + delta, base, delta

    def refiner_configuration_dict(self) -> dict[str, object]:
        result = asdict(self.refiner_configuration)
        result["dilation_schedule"] = list(self.refiner_configuration.dilation_schedule)
        return result

