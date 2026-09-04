"""Stage C auditory-aware loss adapter for conditioned FiLM-SIREN."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch

from mcar.losses import (
    spectral_band_ild_smooth_l1_and_mae,
    strict_hrir_ild_mae,
)
from mcar.training.train_mlp_v2 import (
    LossMetrics, calculate_losses, weighted_direction_mean,
)


@dataclass(frozen=True)
class StageCLossConfiguration:
    erb_weight: float = 0.75
    high_frequency_weight: float = 0.25
    strict_ild_weight: float = 0.75
    spectral_band_ild_weight: float = 0.05
    spectral_band_ild_minimum_center_hz: float = 200.0
    spectral_band_ild_maximum_center_hz: float = 18000.0
    spectral_band_ild_beta_db: float = 0.5
    high_frequency_first_difference_weight: float = 0.0
    high_frequency_second_difference_weight: float = 0.0
    spectral_difference_minimum_frequency_hz: float = 4000.0
    notch_depth_weight: float = 0.0
    notch_minimum_frequency_hz: float = 4000.0
    notch_maximum_frequency_hz: float = 18000.0
    notch_radii_bins: tuple[int, ...] = (4, 8, 16)
    notch_depth_threshold_db: float = 1.0
    notch_softplus_temperature_db: float = 0.5
    lsd_weight: float = 0.0
    lsd_epsilon_db: float = 1e-6

    def validate(self) -> None:
        scalar_weights = (
            self.erb_weight,
            self.high_frequency_weight,
            self.strict_ild_weight,
            self.spectral_band_ild_weight,
            self.high_frequency_first_difference_weight,
            self.high_frequency_second_difference_weight,
            self.notch_depth_weight,
            self.lsd_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in scalar_weights):
            raise ValueError("Stage C loss weights must be finite and non-negative")
        if not math.isfinite(self.lsd_epsilon_db) or self.lsd_epsilon_db <= 0.0:
            raise ValueError("lsd_epsilon_db must be finite and positive")
        if self.spectral_band_ild_beta_db <= 0.0:
            raise ValueError("spectral_band_ild_beta_db must be positive")
        if not self.notch_radii_bins or any(
            radius < 1 for radius in self.notch_radii_bins
        ):
            raise ValueError("notch_radii_bins must contain positive integers")


def spectral_lsd_db(
    error_db: torch.Tensor,
    direction_weights: torch.Tensor,
    epsilon_db: float = 0.0,
) -> torch.Tensor:
    """Frequency RMS per ear/direction, then solid-angle and ear means.

    Use a positive epsilon for backpropagation at zero error; reporting uses
    detached errors with epsilon zero, matching secondary_metrics.full_sphere_lsd.
    """
    return weighted_direction_mean(
        torch.sqrt(torch.mean(error_db.square(), dim=-1) + epsilon_db ** 2),
        direction_weights,
    )


def strict_metadata_to_device(
    metadata: Mapping[str, np.ndarray | int],
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    """Move the frozen strict-HRIR reconstruction metadata to a device."""
    output: dict[str, torch.Tensor | int] = {}
    for name, value in metadata.items():
        output[name] = (
            torch.from_numpy(value).to(device, non_blocking=True)
            if isinstance(value, np.ndarray)
            else int(value)
        )
    return output


def calculate_stage_c_losses(
    global_prediction_normalized: torch.Tensor,
    global_target_normalized: torch.Tensor,
    global_target_db: torch.Tensor,
    global_mca_db: torch.Tensor,
    global_direction_features: torch.Tensor,
    horizontal_prediction_normalized: torch.Tensor,
    horizontal_target_db: torch.Tensor,
    horizontal_mca_db: torch.Tensor,
    horizontal_direction_features: torch.Tensor,
    horizontal_strict_metadata: Mapping[str, torch.Tensor | int],
    frequency_hz: torch.Tensor,
    log_erb_weights: torch.Tensor,
    erb_center_frequency_hz: torch.Tensor,
    target_mean: float,
    target_std: float,
    configuration: StageCLossConfiguration,
    calculate_spectral_diagnostics: bool = False,
) -> tuple[torch.Tensor, LossMetrics]:
    """Combine global auditory losses with an independent horizontal ILD batch."""
    configuration.validate()
    global_loss, metrics = calculate_losses(
        global_prediction_normalized,
        global_target_normalized,
        global_target_db,
        global_mca_db,
        global_direction_features,
        frequency_hz,
        log_erb_weights,
        target_mean,
        target_std,
        configuration.erb_weight,
        configuration.high_frequency_weight,
        0.0,
        True,
        high_frequency_first_difference_weight=(
            configuration.high_frequency_first_difference_weight
        ),
        high_frequency_second_difference_weight=(
            configuration.high_frequency_second_difference_weight
        ),
        spectral_difference_minimum_frequency_hz=(
            configuration.spectral_difference_minimum_frequency_hz
        ),
        notch_depth_weight=configuration.notch_depth_weight,
        notch_minimum_frequency_hz=configuration.notch_minimum_frequency_hz,
        notch_maximum_frequency_hz=configuration.notch_maximum_frequency_hz,
        notch_radii_bins=configuration.notch_radii_bins,
        notch_depth_threshold_db=configuration.notch_depth_threshold_db,
        notch_softplus_temperature_db=(
            configuration.notch_softplus_temperature_db
        ),
        calculate_spectral_diagnostics=calculate_spectral_diagnostics,
    )

    horizontal_prediction_db = (
        horizontal_prediction_normalized.float() * target_std + target_mean
    )
    corrected_db = horizontal_mca_db.float() + horizontal_prediction_db
    reference_db = horizontal_mca_db.float() + horizontal_target_db.float()
    direction_weights = horizontal_direction_features[:, 5].float()
    strict_ild = strict_hrir_ild_mae(
        corrected_db,
        direction_weights,
        horizontal_strict_metadata,
    )
    band_smooth_l1, band_mae = spectral_band_ild_smooth_l1_and_mae(
        corrected_db,
        reference_db,
        direction_weights,
        log_erb_weights,
        erb_center_frequency_hz,
        configuration.spectral_band_ild_minimum_center_hz,
        configuration.spectral_band_ild_maximum_center_hz,
        configuration.spectral_band_ild_beta_db,
    )
    total = global_loss + (
        configuration.strict_ild_weight * strict_ild
        + configuration.spectral_band_ild_weight * band_smooth_l1
    ) / target_std
    if configuration.lsd_weight > 0.0 or calculate_spectral_diagnostics:
        error_db = (
            global_prediction_normalized.float() * target_std + target_mean
            - global_target_db.float()
        )
        global_weights = global_direction_features[:, 5].float()
        lsd_loss = spectral_lsd_db(
            error_db, global_weights, configuration.lsd_epsilon_db
        )
        if configuration.lsd_weight > 0.0:
            total = total + configuration.lsd_weight * lsd_loss / target_std
        metrics.full_sphere_lsd_db = float(
            spectral_lsd_db(error_db.detach(), global_weights).item()
        )
        metrics.lsd_loss_db = float(lsd_loss.detach().item())
    metrics.total = float(total.detach().item())
    metrics.ild_mae_db = float(strict_ild.detach().item())
    metrics.spectral_band_ild_smooth_l1_db = float(
        band_smooth_l1.detach().item()
    )
    metrics.spectral_band_ild_mae_db = float(band_mae.detach().item())
    return total, metrics
