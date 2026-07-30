"""Differentiable losses shared by residual-learning experiments."""

from __future__ import annotations

from typing import Mapping

import torch


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
