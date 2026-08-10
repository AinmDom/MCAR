"""Lightweight binaural spectral CNN stacked on the v2 residual MLP."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mcar.models.residual_mlp import ResidualMLP


def _group_count(channel_count: int, preferred: int = 8) -> int:
    """Return the largest useful GroupNorm divisor not exceeding preferred."""
    for group_count in range(min(preferred, channel_count), 0, -1):
        if channel_count % group_count == 0:
            return group_count
    return 1


class FiLMDepthwiseResidualBlock(nn.Module):
    """Dilated frequency block with direction-conditioned FiLM modulation."""

    def __init__(
        self,
        channels: int,
        condition_features: int,
        expansion: int = 2,
        kernel_size: int = 7,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        hidden_channels = channels * expansion
        padding = dilation * (kernel_size - 1) // 2
        self.expand = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.expand_norm = nn.GroupNorm(
            _group_count(hidden_channels), hidden_channels
        )
        self.frequency_padding = nn.ReflectionPad1d(padding)
        self.frequency_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.frequency_norm = nn.GroupNorm(
            _group_count(hidden_channels), hidden_channels
        )
        self.project = nn.Conv1d(hidden_channels, channels, kernel_size=1)
        self.project_norm = nn.GroupNorm(_group_count(channels), channels)
        self.film = nn.Linear(condition_features, 2 * channels)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
        self, values: torch.Tensor, direction_condition: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.activation(self.expand_norm(self.expand(values)))
        hidden = self.frequency_padding(hidden)
        hidden = self.activation(
            self.frequency_norm(self.frequency_conv(hidden))
        )
        hidden = self.project_norm(self.project(hidden))
        scale, shift = self.film(direction_condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.activation(values + hidden)


class BinauralSpectralCNN(nn.Module):
    """Predict left/right residual refinements from a complete paired spectrum."""

    def __init__(
        self,
        input_channels: int = 7,
        channels: int = 48,
        dilation_schedule: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 7,
        condition_width: int = 64,
    ) -> None:
        super().__init__()
        stem_padding = (kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.ReflectionPad1d(stem_padding),
            nn.Conv1d(input_channels, channels, kernel_size=kernel_size),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, condition_width),
            nn.SiLU(),
            nn.Linear(condition_width, condition_width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            FiLMDepthwiseResidualBlock(
                channels=channels,
                condition_features=condition_width,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for dilation in dilation_schedule
        )
        self.output = nn.Conv1d(channels, 2, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, spectrum_features: torch.Tensor, direction_xyz: torch.Tensor
    ) -> torch.Tensor:
        delta, _ = self.forward_with_features(spectrum_features, direction_xyz)
        return delta

    def forward_with_features(
        self, spectrum_features: torch.Tensor, direction_xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the local delta and its final hidden representation."""
        if spectrum_features.ndim != 3:
            raise ValueError(
                "spectrum_features must have shape [direction, channel, frequency]"
            )
        if direction_xyz.shape != (spectrum_features.shape[0], 3):
            raise ValueError(
                "direction_xyz must have shape [direction, 3]"
            )
        condition = self.direction_encoder(direction_xyz)
        hidden = self.stem(spectrum_features)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden), hidden


class PreNormSpectralAttentionBlock(nn.Module):
    """Small pre-norm transformer block operating on downsampled frequencies."""

    def __init__(self, width: int, head_count: int, expansion: int = 2) -> None:
        super().__init__()
        if width % head_count != 0:
            raise ValueError("attention width must be divisible by head_count")
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            head_count,
            dropout=0.0,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.SiLU(),
            nn.Linear(width * expansion, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feed_forward(self.feed_forward_norm(tokens))


class GlobalSpectralAttention(nn.Module):
    """Model full-spectrum context on a low-resolution frequency sequence."""

    def __init__(
        self,
        input_channels: int,
        local_channels: int,
        width: int = 32,
        head_count: int = 4,
        block_count: int = 2,
        downsample_stride: int = 4,
    ) -> None:
        super().__init__()
        if downsample_stride < 1:
            raise ValueError("downsample_stride must be positive")
        self.downsample_stride = downsample_stride
        self.downsample = nn.Conv1d(
            input_channels,
            width,
            kernel_size=9,
            stride=downsample_stride,
            padding=4,
        )
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            PreNormSpectralAttentionBlock(width, head_count)
            for _ in range(block_count)
        )
        self.output_norm = nn.LayerNorm(width)
        self.to_local_features = nn.Sequential(
            nn.Conv1d(width, local_channels, kernel_size=1),
            nn.GroupNorm(_group_count(local_channels), local_channels),
            nn.SiLU(),
        )
        self.output = nn.Conv1d(local_channels, 2, kernel_size=1)
        # Exact functional compatibility with the source v3.2 checkpoint.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        spectrum_features: torch.Tensor,
        direction_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequency_count = spectrum_features.shape[-1]
        tokens = self.downsample(spectrum_features).transpose(1, 2)
        tokens = tokens + self.direction_encoder(direction_xyz).unsqueeze(1)
        for block in self.blocks:
            tokens = block(tokens)
        hidden = self.output_norm(tokens).transpose(1, 2)
        hidden = F.interpolate(
            hidden,
            size=frequency_count,
            mode="linear",
            align_corners=False,
        )
        hidden = self.to_local_features(hidden)
        return self.output(hidden), hidden


class ResidualMLPCNN(nn.Module):
    """Fuse the v2 pointwise MLP with a binaural spectral delta CNN.

    Input layout is [direction, ear=2, frequency, point_feature=7]. The point
    feature order is the existing residual-learning order:
    normalized MCA, normalized correction, x, y, z, normalized log-frequency,
    and ear flag.
    """

    def __init__(
        self,
        mlp_width: int = 128,
        mlp_block_count: int = 3,
        cnn_channels: int = 48,
        dilation_schedule: tuple[int, ...] = (1, 2, 4, 8),
        global_context_attention: bool = False,
        global_attention_width: int = 32,
        global_attention_heads: int = 4,
        global_attention_blocks: int = 2,
        global_attention_stride: int = 4,
    ) -> None:
        super().__init__()
        self.mlp = ResidualMLP(
            width=mlp_width,
            block_count=mlp_block_count,
        )
        self.cnn = BinauralSpectralCNN(
            channels=cnn_channels,
            dilation_schedule=dilation_schedule,
        )
        self.global_context = (
            GlobalSpectralAttention(
                input_channels=7,
                local_channels=cnn_channels,
                width=global_attention_width,
                head_count=global_attention_heads,
                block_count=global_attention_blocks,
                downsample_stride=global_attention_stride,
            )
            if global_context_attention
            else None
        )
        self.global_gate = (
            nn.Conv1d(2 * cnn_channels, 2, kernel_size=1)
            if global_context_attention
            else None
        )
        if self.global_gate is not None:
            nn.init.zeros_(self.global_gate.weight)
            nn.init.constant_(self.global_gate.bias, -2.0)

    def load_mlp_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        self.mlp.load_state_dict(state_dict, strict=strict)

    def freeze_mlp(self) -> None:
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)
        self.mlp.eval()

    def unfreeze_mlp(self) -> None:
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "ResidualMLPCNN":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.mlp.parameters()):
            self.mlp.eval()
        return self

    def forward(
        self, point_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if point_features.ndim != 4:
            raise ValueError(
                "point_features must have shape [direction, ear, frequency, feature]"
            )
        direction_count, ear_count, frequency_count, feature_count = (
            point_features.shape
        )
        if ear_count != 2 or feature_count != 7:
            raise ValueError(
                f"expected ear=2 and feature=7, got {point_features.shape}"
            )

        base_prediction = self.mlp(
            point_features.reshape(-1, feature_count)
        ).reshape(direction_count, ear_count, frequency_count)

        left = point_features[:, 0]
        right = point_features[:, 1]
        normalized_log_frequency = left[:, :, 5]
        spectrum_features = torch.stack(
            (
                left[:, :, 0],
                left[:, :, 1],
                base_prediction[:, 0],
                right[:, :, 0],
                right[:, :, 1],
                base_prediction[:, 1],
                normalized_log_frequency,
            ),
            dim=1,
        )
        direction_xyz = left[:, 0, 2:5]
        if self.global_context is None or self.global_gate is None:
            delta_prediction = self.cnn(spectrum_features, direction_xyz)
        else:
            local_delta, local_hidden = self.cnn.forward_with_features(
                spectrum_features,
                direction_xyz,
            )
            global_delta, global_hidden = self.global_context(
                spectrum_features,
                direction_xyz,
            )
            gate = torch.sigmoid(
                self.global_gate(
                    torch.cat((local_hidden, global_hidden), dim=1)
                )
            )
            delta_prediction = local_delta + gate * global_delta
        final_prediction = base_prediction + delta_prediction
        return final_prediction, base_prediction, delta_prediction


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
