"""Frequency/source-position conditioned autoencoder for HRTF upsampling.

This module is an adapted implementation of FSP-AE from:

    Y. Ito, T. Nakamura, S. Koyama, S. Sakamoto, and H. Saruwatari,
    "Spatial Upsampling of Head-Related Transfer Function Using Neural
    Network Conditioned on Source Position and Frequency," IEEE Open Journal
    of Signal Processing, vol. 6, pp. 1109--1123, 2025.

The upstream implementation is available at https://github.com/ikets/FSP-AE
under CC BY 4.0. The layer names and tensor layout are intentionally retained
so that published upstream checkpoints can be loaded without key conversion.
Project-specific training and 44.1 kHz reconstruction live outside this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FSPAEConfig:
    """Published FSP-AE v1 architecture defaults."""

    encoder_num_layers: int = 2
    encoder_mid_dim: int = 16
    encoder_out_dim: int = 64
    decoder_num_layers: int = 2
    decoder_mid_dim: int = 16
    decoder_in_dim: int = 64
    weight_bias_generator_num_layers: int = 2
    weight_bias_generator_mid_dim: int = 64
    source_position_fourier_features: int = 16
    frequency_fourier_features: int = 8
    trainable_fourier_features: bool = True
    encoder_use_frequency: bool = True
    decoder_use_frequency: bool = True
    encoder_nonlinear: bool = True
    decoder_nonlinear: bool = True
    frequency_normalization_hz: float = 16_000.0
    radius_normalization_m: float = 1.5
    measurement_count_normalization: float = 865.0

    def __post_init__(self) -> None:
        if self.encoder_out_dim != self.decoder_in_dim:
            raise ValueError("encoder_out_dim must equal decoder_in_dim")
        integer_values = (
            self.encoder_num_layers,
            self.encoder_mid_dim,
            self.encoder_out_dim,
            self.decoder_num_layers,
            self.decoder_mid_dim,
            self.decoder_in_dim,
            self.weight_bias_generator_num_layers,
            self.weight_bias_generator_mid_dim,
            self.source_position_fourier_features,
            self.frequency_fourier_features,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("all FSP-AE architecture dimensions must be positive")


class ResLinearBlock(nn.Module):
    """Mish residual block used inside the weight/bias generators."""

    def __init__(self, channel: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Linear(channel, channel),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(channel, channel),
        )
        self.layers2 = nn.Sequential(nn.Mish(), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers2(self.layers1(values) + values)


class HyperLinear(nn.Module):
    """Generate an affine transform from a conditioning vector."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        input_size: int = 3,
        ch_hidden: int = 32,
        num_hidden: int = 1,
        use_res: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.weight_layers = self._make_generator(
            input_size,
            ch_hidden,
            num_hidden,
            ch_out * ch_in,
            use_res,
            dropout,
        )
        self.bias_layers = self._make_generator(
            input_size,
            ch_hidden,
            num_hidden,
            ch_out,
            use_res,
            dropout,
        )

    @staticmethod
    def _make_generator(
        input_size: int,
        hidden_size: int,
        hidden_count: int,
        output_size: int,
        use_residual: bool,
        dropout: float,
    ) -> nn.Sequential:
        modules: list[nn.Module] = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Dropout(dropout),
        ]
        if use_residual:
            modules.extend(
                ResLinearBlock(hidden_size)
                for _ in range(round(hidden_count / 2))
            )
        else:
            for _ in range(hidden_count):
                modules.extend(
                    (
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.Mish(),
                        nn.Dropout(dropout),
                    )
                )
        modules.append(nn.Linear(hidden_size, output_size))
        return nn.Sequential(*modules)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, conditioning = inputs
        batch_shape = list(values.shape[:-1])
        flattened_count = math.prod(batch_shape)
        weight = self.weight_layers(conditioning).reshape(
            flattened_count, self.ch_out, self.ch_in
        )
        bias = self.bias_layers(conditioning)
        transformed = torch.matmul(
            weight, values.reshape(flattened_count, -1, 1)
        ).reshape(batch_shape + [-1])
        return transformed + bias, conditioning


class HyperLinearBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 32,
        num_hidden: int = 1,
        dropout: float = 0.0,
        use_res: bool = True,
        cond_dim: int = 3,
        post_process: bool = True,
    ) -> None:
        super().__init__()
        self.hyperlinear = HyperLinear(
            in_dim,
            out_dim,
            ch_hidden=hidden_dim,
            num_hidden=num_hidden,
            use_res=use_res,
            input_size=cond_dim,
            dropout=dropout,
        )
        self.layers_post = (
            nn.Sequential(nn.LayerNorm(out_dim), nn.Mish(), nn.Dropout(dropout))
            if post_process
            else nn.Sequential(nn.Identity())
        )

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, conditioning = self.hyperlinear(inputs)
        return self.layers_post(values), conditioning


class FourierFeatureMapping(nn.Module):
    def __init__(
        self, num_features: int, input_dim: int, trainable: bool = True
    ) -> None:
        super().__init__()
        vectors = torch.distributions.MultivariateNormal(
            torch.zeros(input_dim), torch.eye(input_dim)
        ).sample((num_features,))
        self.num_features = num_features
        self.input_dim = input_dim
        if trainable:
            self.v = nn.Parameter(vectors)
        else:
            self.register_buffer("v", vectors)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output_shape = list(values.shape)
        output_shape[-1] = self.num_features
        flattened = values.reshape(-1, self.input_dim).permute(1, 0)
        projected = torch.matmul(self.v, flattened)
        projected = projected.permute(1, 0).reshape(output_shape)
        return torch.cat(
            (
                torch.sin(2 * np.pi * projected),
                torch.cos(2 * np.pi * projected),
            ),
            dim=-1,
        )


class FreqSrcPosCondAutoEncoder(nn.Module):
    """Published FSP-AE model with separately callable encoder and decoder."""

    def __init__(self, config: FSPAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or FSPAEConfig()
        self.stats: dict[str, dict[str, dict[str, torch.Tensor | float]]] = {
            "none": {
                "hrtf_mag": {"mean": 0.0, "std": 1.0},
                "itd": {"mean": 0.0, "std": 1.0},
            }
        }
        self.ffm_srcpos = FourierFeatureMapping(
            self.config.source_position_fourier_features,
            input_dim=3,
            trainable=self.config.trainable_fourier_features,
        )
        self.ffm_freq = FourierFeatureMapping(
            self.config.frequency_fourier_features,
            input_dim=1,
            trainable=self.config.trainable_fourier_features,
        )
        self.radius_norm = self.config.radius_normalization_m
        self.freq_norm = self.config.frequency_normalization_hz
        self.num_mes_norm = self.config.measurement_count_normalization

        encoder_condition_dim = (
            2 * self.config.source_position_fourier_features + 2
        )
        if self.config.encoder_use_frequency:
            encoder_condition_dim += 2 * self.config.frequency_fourier_features
        encoder_modules: list[nn.Module] = []
        for layer_index in range(self.config.encoder_num_layers):
            input_dim = 1 if layer_index == 0 else self.config.encoder_mid_dim
            output_dim = (
                self.config.encoder_out_dim
                if layer_index == self.config.encoder_num_layers - 1
                else self.config.encoder_mid_dim
            )
            encoder_modules.append(
                HyperLinearBlock(
                    in_dim=input_dim,
                    out_dim=output_dim,
                    hidden_dim=self.config.weight_bias_generator_mid_dim,
                    num_hidden=self.config.weight_bias_generator_num_layers,
                    cond_dim=encoder_condition_dim,
                    post_process=(
                        self.config.encoder_nonlinear
                        and layer_index != self.config.encoder_num_layers - 1
                    ),
                )
            )
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_condition_dim = (
            2 * self.config.source_position_fourier_features + 1
        )
        if self.config.decoder_use_frequency:
            decoder_condition_dim += 2 * self.config.frequency_fourier_features
        decoder_modules: list[nn.Module] = []
        for layer_index in range(self.config.decoder_num_layers):
            input_dim = (
                self.config.decoder_in_dim
                if layer_index == 0
                else self.config.decoder_mid_dim
            )
            output_dim = (
                1
                if layer_index == self.config.decoder_num_layers - 1
                else self.config.decoder_mid_dim
            )
            decoder_modules.append(
                HyperLinearBlock(
                    in_dim=input_dim,
                    out_dim=output_dim,
                    hidden_dim=self.config.weight_bias_generator_mid_dim,
                    num_hidden=self.config.weight_bias_generator_num_layers,
                    cond_dim=decoder_condition_dim,
                    post_process=(
                        self.config.decoder_nonlinear
                        and layer_index != self.config.decoder_num_layers - 1
                    ),
                )
            )
        self.decoder = nn.Sequential(*decoder_modules)

    def set_stats(
        self,
        mean: torch.Tensor | float,
        std: torch.Tensor | float,
        dataset_name: str,
        data_type: str = "hrtf_mag",
    ) -> None:
        if float(torch.as_tensor(std)) <= 0:
            raise ValueError("normalization standard deviation must be positive")
        self.stats.setdefault(dataset_name, {})[data_type] = {
            "mean": mean,
            "std": std,
        }

    def _standardize(
        self,
        values: torch.Tensor,
        dataset_name: str,
        data_type: str,
        reverse: bool = False,
    ) -> torch.Tensor:
        if dataset_name not in self.stats or data_type not in self.stats[dataset_name]:
            raise KeyError(f"missing {dataset_name}/{data_type} normalization")
        statistics = self.stats[dataset_name][data_type]
        mean = torch.as_tensor(
            statistics["mean"], dtype=values.dtype, device=values.device
        )
        std = torch.as_tensor(
            statistics["std"], dtype=values.dtype, device=values.device
        )
        return values * std + mean if reverse else (values - mean) / std

    def get_conditioning_vector(
        self,
        positions_cartesian: torch.Tensor,
        frequency_hz: torch.Tensor,
        use_frequency: bool,
        use_measurement_count: bool,
    ) -> torch.Tensor:
        subject_count, position_count, _ = positions_cartesian.shape
        frequency_count = frequency_hz.shape[1]
        device = positions_cartesian.device
        dtype = positions_cartesian.dtype
        parts: list[torch.Tensor] = []

        normalized_positions = positions_cartesian / self.radius_norm
        mirror = torch.tensor(
            [1.0, -1.0, 1.0], device=device, dtype=dtype
        )
        mirrored_positions = normalized_positions * mirror[None, None, :]
        position_features = self.ffm_srcpos(normalized_positions)
        position_features = position_features.unsqueeze(2).tile(
            1, 1, frequency_count, 1
        )
        mirrored_features = self.ffm_srcpos(mirrored_positions)
        mirrored_features = mirrored_features.unsqueeze(2).tile(
            1, 1, frequency_count, 1
        )
        parts.append(
            torch.cat(
                (
                    position_features,
                    mirrored_features,
                    position_features[:, :, 0:1, :],
                ),
                dim=2,
            )
        )

        if use_frequency:
            frequency_features = self.ffm_freq(
                (frequency_hz / self.freq_norm).unsqueeze(-1)
            )
            frequency_features = frequency_features.reshape(
                subject_count, 1, frequency_count, -1
            ).tile(1, position_count, 2, 1)
            frequency_features = torch.cat(
                (
                    frequency_features,
                    torch.zeros(
                        subject_count,
                        position_count,
                        1,
                        frequency_features.shape[-1],
                        device=device,
                        dtype=dtype,
                    ),
                ),
                dim=2,
            )
            parts.append(frequency_features)

        if use_measurement_count:
            parts.append(
                torch.full(
                    (subject_count, position_count, 2 * frequency_count + 1, 1),
                    position_count / self.num_mes_norm,
                    device=device,
                    dtype=dtype,
                )
            )
        input_type = torch.cat(
            (
                torch.zeros(2 * frequency_count, device=device, dtype=dtype),
                torch.ones(1, device=device, dtype=dtype),
            )
        ).reshape(1, 1, 2 * frequency_count + 1, 1)
        parts.append(input_type.tile(subject_count, position_count, 1, 1))
        return torch.cat(parts, dim=-1)

    def encode(
        self,
        hrtf_magnitude_db: torch.Tensor,
        itd_seconds: torch.Tensor,
        frequency_hz: torch.Tensor,
        measurement_positions_cartesian: torch.Tensor,
        dataset_name: str = "none",
    ) -> torch.Tensor:
        if hrtf_magnitude_db.ndim != 4 or hrtf_magnitude_db.shape[2] != 2:
            raise ValueError("hrtf_magnitude_db must have shape [S, B, 2, L]")
        subject_count, measurement_count, _, frequency_count = (
            hrtf_magnitude_db.shape
        )
        if itd_seconds.shape != (subject_count, measurement_count):
            raise ValueError("itd_seconds must have shape [S, B]")
        if frequency_hz.shape != (subject_count, frequency_count):
            raise ValueError("frequency_hz must have shape [S, L]")
        if measurement_positions_cartesian.shape != (
            subject_count,
            measurement_count,
            3,
        ):
            raise ValueError("measurement positions must have shape [S, B, 3]")

        magnitude = self._standardize(
            hrtf_magnitude_db, dataset_name, "hrtf_mag"
        )
        itd = self._standardize(
            itd_seconds, dataset_name, "itd"
        ).unsqueeze(-1)
        magnitude = torch.cat((magnitude[:, :, 0], magnitude[:, :, 1]), dim=-1)
        encoder_input = torch.cat((magnitude, itd), dim=-1).unsqueeze(-1)
        encoder_condition = self.get_conditioning_vector(
            measurement_positions_cartesian,
            frequency_hz,
            use_frequency=self.config.encoder_use_frequency,
            use_measurement_count=True,
        )
        latent = self.encoder((encoder_input, encoder_condition))[0]
        return torch.mean(latent, dim=1, keepdim=True)

    def decode(
        self,
        prototype: torch.Tensor,
        frequency_hz: torch.Tensor,
        target_positions_cartesian: torch.Tensor,
        dataset_name: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        subject_count, target_count, _ = target_positions_cartesian.shape
        frequency_count = frequency_hz.shape[1]
        if prototype.shape[:3] != (subject_count, 1, 2 * frequency_count + 1):
            raise ValueError("prototype shape does not match subjects/frequencies")
        decoder_input = prototype.tile(1, target_count, 1, 1)
        decoder_condition = self.get_conditioning_vector(
            target_positions_cartesian,
            frequency_hz,
            use_frequency=self.config.decoder_use_frequency,
            use_measurement_count=False,
        )
        output = self.decoder((decoder_input, decoder_condition))[0]
        magnitude = torch.cat(
            (
                output[:, :, None, :frequency_count, 0],
                output[:, :, None, frequency_count : 2 * frequency_count, 0],
            ),
            dim=2,
        )
        itd = output[:, :, -1, 0]
        return (
            self._standardize(
                magnitude, dataset_name, "hrtf_mag", reverse=True
            ),
            self._standardize(itd, dataset_name, "itd", reverse=True),
        )

    def forward(
        self,
        hrtf_magnitude_db: torch.Tensor,
        itd_seconds: torch.Tensor,
        frequency_hz: torch.Tensor,
        measurement_positions_cartesian: torch.Tensor,
        target_positions_cartesian: torch.Tensor,
        dataset_name: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototype = self.encode(
            hrtf_magnitude_db,
            itd_seconds,
            frequency_hz,
            measurement_positions_cartesian,
            dataset_name,
        )
        return self.decode(
            prototype,
            frequency_hz,
            target_positions_cartesian,
            dataset_name,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
