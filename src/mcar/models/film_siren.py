"""Q26-conditioned FiLM-SIREN models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from mcar.models.siren import (
    SineLayer,
    reference_bias_bound,
    siren_weight_bound,
)


MODULATION_VARIANTS = {"concat", "amplitude", "phase", "full"}
PLACEMENTS = {"none", "late", "hidden", "all"}


@dataclass(frozen=True)
class ConditionEncoderConfig:
    frequency_count: int = 463
    convolution_channels: int = 32
    direction_embedding_dimension: int = 128
    latent_dimension: int = 128

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class FilmSirenConfig:
    coordinate_dimension: int = 5
    hidden_width: int = 256
    sine_layer_count: int = 6
    first_omega: float = 20.0
    hidden_omega: float = 20.0
    output_dimension: int = 2
    latent_dimension: int = 128
    modulation_variant: str = "phase"
    placement: str = "hidden"
    modulator_width: int = 128
    gamma_max_delta: float = 0.5
    beta_max: float = torch.pi
    amplitude_max_delta: float = 0.5

    def validate(self) -> None:
        if min(
            self.coordinate_dimension,
            self.hidden_width,
            self.sine_layer_count,
            self.output_dimension,
            self.latent_dimension,
            self.modulator_width,
        ) < 1:
            raise ValueError("model dimensions and layer count must be positive")
        if self.first_omega <= 0.0 or self.hidden_omega <= 0.0:
            raise ValueError("SIREN omega values must be positive")
        if self.modulation_variant not in MODULATION_VARIANTS:
            raise ValueError(
                f"modulation_variant must be one of {sorted(MODULATION_VARIANTS)}"
            )
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {sorted(PLACEMENTS)}")
        if self.modulation_variant == "concat" and self.placement != "none":
            raise ValueError("concat uses placement='none'")
        if self.modulation_variant != "concat" and self.placement == "none":
            raise ValueError("FiLM variants require a non-empty placement")
        if min(
            self.gamma_max_delta,
            self.beta_max,
            self.amplitude_max_delta,
        ) <= 0.0:
            raise ValueError("modulation bounds must be positive")


class Q26ConditionEncoder(nn.Module):
    """Encode a direction set of binaural Q26 spectra and xyz coordinates."""

    def __init__(self, configuration: ConditionEncoderConfig) -> None:
        super().__init__()
        configuration.validate()
        self.configuration = configuration
        channels = configuration.convolution_channels
        self.spectral = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=9, stride=2, padding=4),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=9, stride=2, padding=4),
            nn.SiLU(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, 2, configuration.frequency_count)
            flattened = int(self.spectral(probe).numel())
        self.direction_projection = nn.Sequential(
            nn.Linear(flattened + 3, configuration.direction_embedding_dimension),
            nn.SiLU(),
        )
        self.latent_projection = nn.Linear(
            configuration.direction_embedding_dimension,
            configuration.latent_dimension,
        )

    def forward(
        self,
        normalized_magnitude: torch.Tensor,
        xyz: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_magnitude.ndim != 4:
            raise ValueError("normalized_magnitude must have shape [B,2,Q,F]")
        batch, ears, directions, frequencies = normalized_magnitude.shape
        if ears != 2 or frequencies != self.configuration.frequency_count:
            raise ValueError("condition magnitude has incompatible ear/frequency shape")
        if xyz.shape != (batch, directions, 3):
            raise ValueError("xyz must have shape [B,Q,3]")
        if mask.shape != (batch, directions):
            raise ValueError("mask must have shape [B,Q]")
        mask = mask.bool()
        if bool(torch.any(torch.sum(mask, dim=1) == 0).item()):
            raise ValueError("each condition set must contain a valid direction")

        spectra = normalized_magnitude.permute(0, 2, 1, 3).reshape(
            batch * directions,
            2,
            frequencies,
        )
        encoded = self.spectral(spectra).flatten(start_dim=1)
        encoded = torch.cat((encoded, xyz.reshape(batch * directions, 3)), dim=1)
        encoded = self.direction_projection(encoded).reshape(batch, directions, -1)
        weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = torch.sum(encoded * weights, dim=1) / torch.sum(
            weights,
            dim=1,
        ).clamp_min(1.0)
        return self.latent_projection(pooled)


def placement_layer_indices(configuration: FilmSirenConfig) -> tuple[int, ...]:
    """Return zero-based sine-layer indices for the named placement."""
    count = configuration.sine_layer_count
    if configuration.placement == "none":
        return ()
    if configuration.placement == "all":
        return tuple(range(count))
    if configuration.placement == "hidden":
        return tuple(range(1, count))
    if configuration.placement == "late":
        return tuple(range(count // 2, count))
    raise AssertionError("configuration validation should reject this placement")


class LayerModulator(nn.Module):
    """Shared latent trunk with separate zero-initialized per-layer heads."""

    def __init__(self, configuration: FilmSirenConfig) -> None:
        super().__init__()
        self.configuration = configuration
        self.layer_indices = placement_layer_indices(configuration)
        self.trunk = nn.Sequential(
            nn.Linear(configuration.latent_dimension, configuration.modulator_width),
            nn.SiLU(),
        )
        parameter_count = {
            "amplitude": 1,
            "phase": 2,
            "full": 3,
        }[configuration.modulation_variant]
        self.parameter_count = parameter_count
        self.heads = nn.ModuleDict(
            {
                str(index): nn.Linear(
                    configuration.modulator_width,
                    configuration.hidden_width * parameter_count,
                )
                for index in self.layer_indices
            }
        )
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, latent: torch.Tensor) -> dict[int, tuple[torch.Tensor, ...]]:
        shared = self.trunk(latent)
        return {
            index: tuple(
                self.heads[str(index)](shared).chunk(self.parameter_count, dim=-1)
            )
            for index in self.layer_indices
        }


class FilmSiren(nn.Module):
    """Frozen Stage-A SIREN architecture with subject conditioning."""

    def __init__(
        self,
        configuration: FilmSirenConfig,
        encoder_configuration: ConditionEncoderConfig,
    ) -> None:
        super().__init__()
        configuration.validate()
        encoder_configuration.validate()
        if encoder_configuration.latent_dimension != configuration.latent_dimension:
            raise ValueError("encoder and FiLM latent dimensions must match")
        self.configuration = configuration
        self.encoder_configuration = encoder_configuration

        first_input = configuration.coordinate_dimension
        if configuration.modulation_variant == "concat":
            first_input += configuration.latent_dimension
        layers: list[SineLayer] = [
            SineLayer(
                first_input,
                configuration.hidden_width,
                omega=configuration.first_omega,
                first=True,
            )
        ]
        for _ in range(configuration.sine_layer_count - 1):
            layers.append(
                SineLayer(
                    configuration.hidden_width,
                    configuration.hidden_width,
                    omega=configuration.hidden_omega,
                    first=False,
                )
            )
        self.sine_layers = nn.ModuleList(layers)
        self.output = nn.Linear(
            configuration.hidden_width,
            configuration.output_dimension,
        )
        self._reset_output_parameters()

        self.condition_encoder = Q26ConditionEncoder(encoder_configuration)
        self.modulator = (
            None
            if configuration.modulation_variant == "concat"
            else LayerModulator(configuration)
        )

    def _reset_output_parameters(self) -> None:
        bound = siren_weight_bound(
            self.configuration.hidden_width,
            self.configuration.hidden_omega,
            first=False,
        )
        bias_bound = reference_bias_bound(self.configuration.hidden_width)
        with torch.no_grad():
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.uniform_(-bias_bound, bias_bound)

    def encode_condition(
        self,
        normalized_magnitude: torch.Tensor,
        xyz: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.condition_encoder(normalized_magnitude, xyz, mask)

    @torch.no_grad()
    def modulation_statistics(self, latent: torch.Tensor) -> dict[str, object]:
        """Summarize transformed modulation parameters for a latent batch."""
        variant = self.configuration.modulation_variant
        if variant == "concat":
            return {"variant": variant, "saturation_threshold": 0.95, "layers": {}}
        if self.modulator is None:
            raise AssertionError("FiLM variant is missing its modulator")
        parameters = self.modulator(latent)
        names = {
            "amplitude": ("amplitude",),
            "phase": ("gamma", "beta"),
            "full": ("gamma", "beta", "amplitude"),
        }[variant]
        layers: dict[str, object] = {}
        for layer_index, raw_parameters in parameters.items():
            layer: dict[str, object] = {}
            for name, raw in zip(names, raw_parameters):
                bounded = torch.tanh(raw.float())
                if name == "gamma":
                    transformed = 1.0 + self.configuration.gamma_max_delta * bounded
                elif name == "beta":
                    transformed = self.configuration.beta_max * bounded
                else:
                    transformed = (
                        1.0
                        + self.configuration.amplitude_max_delta * bounded
                    )
                layer[name] = {
                    "mean": float(torch.mean(transformed).item()),
                    "std_population": float(torch.std(transformed, unbiased=False).item()),
                    "minimum": float(torch.min(transformed).item()),
                    "maximum": float(torch.max(transformed).item()),
                    "saturation_fraction": float(
                        torch.mean((torch.abs(bounded) > 0.95).float()).item()
                    ),
                }
            layers[str(layer_index + 1)] = layer
        return {
            "variant": variant,
            "saturation_threshold": 0.95,
            "layers": layers,
        }

    def forward(self, coordinates: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        squeeze_batch = coordinates.ndim == 2
        if squeeze_batch:
            coordinates = coordinates.unsqueeze(0)
        if coordinates.ndim != 3:
            raise ValueError("coordinates must have shape [N,C] or [B,N,C]")
        batch, queries, coordinate_dimension = coordinates.shape
        if coordinate_dimension != self.configuration.coordinate_dimension:
            raise ValueError("coordinate feature dimension does not match configuration")
        if latent.shape != (batch, self.configuration.latent_dimension):
            raise ValueError("latent must have shape [B,latent_dimension]")

        if self.configuration.modulation_variant == "concat":
            expanded = latent[:, None, :].expand(-1, queries, -1)
            hidden = torch.cat((coordinates, expanded), dim=-1)
            for layer in self.sine_layers:
                hidden = layer(hidden)
        else:
            if self.modulator is None:
                raise AssertionError("FiLM variant is missing its modulator")
            parameters = self.modulator(latent)
            hidden = coordinates
            for index, layer in enumerate(self.sine_layers):
                preactivation = layer.linear(hidden)
                if index not in parameters:
                    hidden = torch.sin(layer.omega * preactivation)
                    continue
                raw = parameters[index]
                variant = self.configuration.modulation_variant
                if variant == "amplitude":
                    (raw_amplitude,) = raw
                    activation = torch.sin(layer.omega * preactivation)
                    amplitude = 1.0 + self.configuration.amplitude_max_delta * torch.tanh(
                        raw_amplitude
                    )
                    hidden = amplitude[:, None, :] * activation
                else:
                    raw_gamma, raw_beta = raw[:2]
                    gamma = 1.0 + self.configuration.gamma_max_delta * torch.tanh(
                        raw_gamma
                    )
                    beta = self.configuration.beta_max * torch.tanh(raw_beta)
                    activation = torch.sin(
                        layer.omega * gamma[:, None, :] * preactivation
                        + beta[:, None, :]
                    )
                    if variant == "full":
                        raw_amplitude = raw[2]
                        amplitude = (
                            1.0
                            + self.configuration.amplitude_max_delta
                            * torch.tanh(raw_amplitude)
                        )
                        activation = amplitude[:, None, :] * activation
                    hidden = activation
        output = self.output(hidden)
        return output.squeeze(0) if squeeze_batch else output

    def forward_from_condition(
        self,
        coordinates: torch.Tensor,
        normalized_magnitude: torch.Tensor,
        xyz: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encode_condition(normalized_magnitude, xyz, mask)
        return self.forward(coordinates, latent)
