"""Small residual MLP used to predict normalized MCA spectral residuals."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, input_features: int = 7, width: int = 128, block_count: int = 3) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_features, width), nn.SiLU())
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(block_count))
        self.output = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
