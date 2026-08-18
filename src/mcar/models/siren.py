"""Coordinate-based sinusoidal representation networks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SirenConfig:
    """Fully specified plain-SIREN architecture.

    ``sine_layer_count`` includes the first sine layer. The output projection
    is an additional linear layer.
    """

    input_dimension: int = 4
    hidden_width: int = 256
    sine_layer_count: int = 6
    first_omega: float = 30.0
    hidden_omega: float = 30.0
    output_dimension: int = 2

    def validate(self) -> None:
        if self.input_dimension < 1:
            raise ValueError("input_dimension must be positive")
        if self.hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        if self.sine_layer_count < 1:
            raise ValueError("sine_layer_count must be positive")
        if self.first_omega <= 0.0 or self.hidden_omega <= 0.0:
            raise ValueError("SIREN omega values must be positive")
        if self.output_dimension < 1:
            raise ValueError("output_dimension must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def siren_weight_bound(
    input_dimension: int,
    omega: float,
    *,
    first: bool,
) -> float:
    """Return the symmetric SIREN weight-initialization bound."""
    if input_dimension < 1:
        raise ValueError("input_dimension must be positive")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if first:
        return 1.0 / input_dimension
    return math.sqrt(6.0 / input_dimension) / omega


def reference_bias_bound(input_dimension: int) -> float:
    """Return the explicit bias bound used by the reference implementation.

    Reference SIREN code replaces ``nn.Linear`` weights after construction but
    retains PyTorch's seeded bias initialization. We spell that initialization
    out to make it independent of a future ``nn.Linear`` implementation.
    """
    if input_dimension < 1:
        raise ValueError("input_dimension must be positive")
    return 1.0 / math.sqrt(input_dimension)


class SineLayer(nn.Module):
    """A linear projection followed by a frequency-scaled sine."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        *,
        omega: float,
        first: bool,
    ) -> None:
        super().__init__()
        if output_dimension < 1:
            raise ValueError("output_dimension must be positive")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.omega = float(omega)
        self.first = bool(first)
        self.linear = nn.Linear(input_dimension, output_dimension)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_bound = siren_weight_bound(
            self.input_dimension,
            self.omega,
            first=self.first,
        )
        bias_bound = reference_bias_bound(self.input_dimension)
        with torch.no_grad():
            self.linear.weight.uniform_(-weight_bound, weight_bound)
            self.linear.bias.uniform_(-bias_bound, bias_bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(inputs))


class Siren(nn.Module):
    """Plain coordinate SIREN with a linear binaural residual output."""

    def __init__(self, configuration: SirenConfig) -> None:
        super().__init__()
        configuration.validate()
        self.configuration = configuration
        layers: list[nn.Module] = [
            SineLayer(
                configuration.input_dimension,
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
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        weight_bound = siren_weight_bound(
            self.configuration.hidden_width,
            self.configuration.hidden_omega,
            first=False,
        )
        bias_bound = reference_bias_bound(self.configuration.hidden_width)
        with torch.no_grad():
            self.output.weight.uniform_(-weight_bound, weight_bound)
            self.output.bias.uniform_(-bias_bound, bias_bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.configuration.input_dimension:
            raise ValueError(
                "Expected coordinate dimension "
                f"{self.configuration.input_dimension}, got "
                f"{coordinates.shape[-1]}"
            )
        hidden = coordinates
        for layer in self.sine_layers:
            hidden = layer(hidden)
        return self.output(hidden)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
