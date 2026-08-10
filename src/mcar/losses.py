"""Differentiable losses shared by residual-learning experiments."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


def multi_scale_notch_depth_mae(
    corrected_db: torch.Tensor,
    reference_db: torch.Tensor,
    direction_weights: torch.Tensor,
    frequency_hz: torch.Tensor,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    radii_bins: tuple[int, ...],
    depth_threshold_db: float,
    softplus_temperature_db: float,
) -> torch.Tensor:
    """Return direction-weighted multi-scale notch-depth-map MAE in dB.

    For each radius, local notch depth is the mean of the two shoulders minus
    the center magnitude. A shifted notch changes the depth map at both its
    reference and predicted locations, while the smooth threshold retains a
    useful gradient for missing notches.
    """
    if corrected_db.shape != reference_db.shape or corrected_db.ndim != 3:
        raise ValueError(
            "corrected_db and reference_db must share shape "
            "[ear,direction,frequency]"
        )
    if direction_weights.shape != (corrected_db.shape[1],):
        raise ValueError("direction_weights must match the direction axis")
    if frequency_hz.shape != (corrected_db.shape[2],):
        raise ValueError("frequency_hz must match the frequency axis")
    if minimum_frequency_hz < 0.0:
        raise ValueError("minimum_frequency_hz must be non-negative")
    if maximum_frequency_hz <= minimum_frequency_hz:
        raise ValueError(
            "maximum_frequency_hz must exceed minimum_frequency_hz"
        )
    if not radii_bins or any(radius < 1 for radius in radii_bins):
        raise ValueError("radii_bins must contain positive integers")
    if len(set(radii_bins)) != len(radii_bins):
        raise ValueError("radii_bins must not contain duplicates")
    if depth_threshold_db < 0.0:
        raise ValueError("depth_threshold_db must be non-negative")
    if softplus_temperature_db <= 0.0:
        raise ValueError("softplus_temperature_db must be positive")
    if bool(torch.any(direction_weights < 0.0).item()):
        raise ValueError("direction_weights must be non-negative")
    if not bool(torch.sum(direction_weights) > 0.0):
        raise ValueError("direction_weights must have a positive sum")

    frequency_hz = frequency_hz.float()
    if not bool(torch.all(torch.diff(frequency_hz) > 0.0).item()):
        raise ValueError("frequency_hz must be strictly increasing")
    normalized_weights = direction_weights.float()
    normalized_weights = normalized_weights / torch.sum(normalized_weights)
    spatial_weights = normalized_weights.view(1, -1, 1)
    scale_errors: list[torch.Tensor] = []

    def notch_depth_map(
        magnitude_db: torch.Tensor,
        radius_bins: int,
        center_mask: torch.Tensor,
    ) -> torch.Tensor:
        shoulder_db = 0.5 * (
            magnitude_db[..., : -2 * radius_bins]
            + magnitude_db[..., 2 * radius_bins :]
        )
        center_db = magnitude_db[..., radius_bins:-radius_bins]
        raw_depth_db = shoulder_db - center_db
        return softplus_temperature_db * nn.functional.softplus(
            (raw_depth_db - depth_threshold_db)
            / softplus_temperature_db
        )[..., center_mask]

    for radius_bins in radii_bins:
        if 2 * radius_bins >= corrected_db.shape[-1]:
            raise ValueError("A notch radius is too large for the frequency axis")
        center_frequency_hz = frequency_hz[radius_bins:-radius_bins]
        center_mask = (
            (center_frequency_hz >= minimum_frequency_hz)
            & (center_frequency_hz <= maximum_frequency_hz)
        )
        if not bool(torch.any(center_mask).item()):
            raise ValueError("No notch center remains inside the requested range")
        corrected_depth_db = notch_depth_map(
            corrected_db.float(), radius_bins, center_mask
        )
        reference_depth_db = notch_depth_map(
            reference_db.float(), radius_bins, center_mask
        )
        absolute_error_db = torch.abs(
            corrected_depth_db - reference_depth_db
        )
        scale_errors.append(
            torch.mean(
                torch.sum(absolute_error_db * spatial_weights, dim=1)
            )
        )
    return torch.mean(torch.stack(scale_errors))


def spectral_band_ild_smooth_l1_and_mae(
    corrected_db: torch.Tensor,
    reference_db: torch.Tensor,
    direction_weights: torch.Tensor,
    log_band_weights: torch.Tensor,
    band_center_frequency_hz: torch.Tensor,
    minimum_band_center_hz: float,
    maximum_band_center_hz: float,
    beta_db: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal spectral-band ILD SmoothL1 and MAE in dB.

    Magnitude spectra are converted to power in each auditory band before
    the left-minus-right ILD is formed. Direction weights are normalized
    independently of the number of retained bands.
    """
    if corrected_db.shape != reference_db.shape or corrected_db.ndim != 3:
        raise ValueError(
            "corrected_db and reference_db must share shape "
            "[ear,direction,frequency]"
        )
    if corrected_db.shape[0] != 2:
        raise ValueError("Spectral-band ILD requires exactly two ears")
    if direction_weights.shape != (corrected_db.shape[1],):
        raise ValueError("direction_weights must match the direction axis")
    if log_band_weights.ndim != 4 or log_band_weights.shape[:2] != (1, 1):
        raise ValueError("log_band_weights must have shape [1,1,band,frequency]")
    if log_band_weights.shape[-1] != corrected_db.shape[-1]:
        raise ValueError("Band weights must match the frequency axis")
    if band_center_frequency_hz.shape != (log_band_weights.shape[-2],):
        raise ValueError("Band-center frequencies must match the band axis")
    if minimum_band_center_hz < 0.0:
        raise ValueError("minimum_band_center_hz must be non-negative")
    if maximum_band_center_hz <= minimum_band_center_hz:
        raise ValueError(
            "maximum_band_center_hz must exceed minimum_band_center_hz"
        )
    if beta_db <= 0.0:
        raise ValueError("beta_db must be positive")
    if bool(torch.any(direction_weights < 0.0).item()):
        raise ValueError("direction_weights must be non-negative")

    band_mask = (
        (band_center_frequency_hz >= minimum_band_center_hz)
        & (band_center_frequency_hz <= maximum_band_center_hz)
    )
    if not bool(torch.any(band_mask).item()):
        raise ValueError("No auditory band remains inside the requested range")

    scale = torch.log(corrected_db.new_tensor(10.0)) / 10.0
    inverse_scale = 1.0 / scale

    def band_energy_db(magnitude_db: torch.Tensor) -> torch.Tensor:
        log_power = (
            scale * magnitude_db.float().unsqueeze(-2)
            + log_band_weights.float()
        )
        return inverse_scale * torch.logsumexp(log_power, dim=-1)

    corrected_band_db = band_energy_db(corrected_db)[..., band_mask]
    reference_band_db = band_energy_db(reference_db)[..., band_mask]
    corrected_ild_db = corrected_band_db[0] - corrected_band_db[1]
    reference_ild_db = reference_band_db[0] - reference_band_db[1]
    ild_error_db = corrected_ild_db - reference_ild_db
    smooth_l1 = nn.functional.smooth_l1_loss(
        corrected_ild_db,
        reference_ild_db,
        beta=beta_db,
        reduction="none",
    )
    normalized_weights = direction_weights.float()
    normalized_weights = normalized_weights / torch.clamp(
        torch.sum(normalized_weights), min=1e-12
    )
    weights = normalized_weights.view(-1, 1)
    smooth_l1_mean = torch.mean(torch.sum(smooth_l1 * weights, dim=0))
    mae = torch.mean(torch.sum(torch.abs(ild_error_db) * weights, dim=0))
    return smooth_l1_mean, mae


def high_frequency_spectral_difference_mae(
    corrected_db: torch.Tensor,
    reference_db: torch.Tensor,
    direction_weights: torch.Tensor,
    frequency_hz: torch.Tensor,
    minimum_frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direction-weighted first/second spectral-shape errors.

    The current Q26 frequency grid is uniformly spaced, so the metrics are
    expressed in dB/bin and dB/bin^2. A difference stencil is retained only
    when all of its frequency samples are at or above the cutoff.
    """
    if corrected_db.shape != reference_db.shape or corrected_db.ndim != 3:
        raise ValueError(
            "corrected_db and reference_db must share shape "
            "[ear,direction,frequency]"
        )
    if direction_weights.shape != (corrected_db.shape[1],):
        raise ValueError("direction_weights must match the direction axis")
    if frequency_hz.shape != (corrected_db.shape[2],):
        raise ValueError("frequency_hz must match the frequency axis")
    if corrected_db.shape[-1] < 3:
        raise ValueError("At least three frequency bins are required")
    if minimum_frequency_hz < 0.0:
        raise ValueError("minimum_frequency_hz must be non-negative")
    frequency_hz = frequency_hz.float()
    if not bool(torch.all(torch.diff(frequency_hz) > 0.0).item()):
        raise ValueError("frequency_hz must be strictly increasing")

    first_mask = frequency_hz[:-1] >= minimum_frequency_hz
    second_mask = frequency_hz[:-2] >= minimum_frequency_hz
    if not bool(torch.any(first_mask).item()):
        raise ValueError("No first-difference stencil remains above cutoff")
    if not bool(torch.any(second_mask).item()):
        raise ValueError("No second-difference stencil remains above cutoff")

    error_db = corrected_db.float() - reference_db.float()
    first_error = torch.abs(torch.diff(error_db, dim=-1)[..., first_mask])
    second_error = torch.abs(
        torch.diff(error_db, n=2, dim=-1)[..., second_mask]
    )
    normalized_weights = direction_weights.float()
    normalized_weights = normalized_weights / torch.clamp(
        torch.sum(normalized_weights), min=1e-12
    )
    weights = normalized_weights.view(1, -1, 1)
    first_mae = torch.mean(torch.sum(first_error * weights, dim=1))
    second_mae = torch.mean(torch.sum(second_error * weights, dim=1))
    return first_mae, second_mae


def strict_hrir_ild_errors(
    corrected_selected_db: torch.Tensor,
    metadata: Mapping[str, torch.Tensor | int],
) -> torch.Tensor:
    """Match the strict MATLAB HRIR-energy ILD definition.

    The network changes only the selected MCA magnitudes. Original MCA phase
    is restored at those bins, all unselected complex bins remain fixed, the
    single-sided spectrum is mirrored, and the IFFT is cropped to the original
    HRIR length before left/right energies are calculated.
    """
    if corrected_selected_db.ndim != 3:
        raise ValueError(
            "corrected_selected_db must have shape [ear,direction,frequency]"
        )
    if corrected_selected_db.shape[0] != 2:
        raise ValueError("Strict ILD requires exactly two ears")

    selected_phase = metadata["mca_selected_phase_rad"]
    outside_real = metadata["mca_outside_real"]
    outside_imag = metadata["mca_outside_imag"]
    selected_indices = metadata["selected_bin_indices_zero_based"]
    outside_indices = metadata["outside_bin_indices_zero_based"]
    reference_ild_db = metadata["reference_ild_db"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            selected_phase,
            outside_real,
            outside_imag,
            selected_indices,
            outside_indices,
            reference_ild_db,
        )
    ):
        raise TypeError("Strict ILD tensor metadata is incomplete")

    selected_phase = selected_phase.float()
    outside_real = outside_real.float()
    outside_imag = outside_imag.float()
    selected_indices = selected_indices.long()
    outside_indices = outside_indices.long()
    reference_ild_db = reference_ild_db.float()
    single_sided_count = int(metadata["single_sided_frequency_count"])
    hrir_length = int(metadata["hrir_length"])

    if selected_phase.shape != corrected_selected_db.shape:
        raise ValueError(
            "Selected MCA phase does not match corrected magnitude shape"
        )
    if outside_real.shape != outside_imag.shape:
        raise ValueError("Outside-band real/imaginary tensors do not match")
    if outside_real.shape[:2] != corrected_selected_db.shape[:2]:
        raise ValueError("Outside-band MCA tensor has incompatible dimensions")
    if selected_indices.numel() != corrected_selected_db.shape[-1]:
        raise ValueError("Selected-bin index count does not match spectrum")
    if outside_indices.numel() != outside_real.shape[-1]:
        raise ValueError("Outside-bin index count does not match spectrum")
    if selected_indices.numel() + outside_indices.numel() != single_sided_count:
        raise ValueError("Strict ILD bin indices do not cover the full spectrum")

    magnitude = torch.pow(10.0, corrected_selected_db.float() / 20.0)
    selected_complex = torch.polar(magnitude, selected_phase)
    outside_complex = torch.complex(outside_real, outside_imag)
    single_sided = torch.zeros(
        (*corrected_selected_db.shape[:2], single_sided_count),
        dtype=selected_complex.dtype,
        device=corrected_selected_db.device,
    )
    single_sided = single_sided.index_copy(
        -1, outside_indices, outside_complex
    )
    single_sided = single_sided.index_copy(
        -1, selected_indices, selected_complex
    )
    both_sided = torch.cat(
        (
            single_sided,
            torch.conj(torch.flip(single_sided[..., 1:-1], dims=(-1,))),
        ),
        dim=-1,
    )
    hrir = torch.fft.ifft(both_sided, dim=-1).real[..., :hrir_length]
    energy = torch.sum(hrir.square(), dim=-1).clamp_min(1e-20)
    predicted_ild_db = 10.0 * torch.log10(energy[0] / energy[1])
    if reference_ild_db.shape != predicted_ild_db.shape:
        raise ValueError("Reference ILD does not match sampled directions")

    return torch.abs(predicted_ild_db - reference_ild_db)


def strict_hrir_ild_mae(
    corrected_selected_db: torch.Tensor,
    direction_weights: torch.Tensor,
    metadata: Mapping[str, torch.Tensor | int],
) -> torch.Tensor:
    """Return the direction-weighted mean strict HRIR-energy ILD error."""
    errors = strict_hrir_ild_errors(corrected_selected_db, metadata)
    normalized_weights = direction_weights.float()
    normalized_weights = normalized_weights / torch.clamp(
        torch.sum(normalized_weights), min=1e-12
    )
    return torch.sum(errors * normalized_weights)
