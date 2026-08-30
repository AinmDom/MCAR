"""Qualified mechanisms, ITD reconstruction, and dominant-notch metrics."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks, savgol_filter

from mcar.evaluation.secondary_metrics import normalized_direction_weights


def enable_external_torchaudio(site_packages: Path) -> str:
    """Expose an exact matching torchaudio installation without changing an env."""
    import torch as _torch  # Import the active runtime before extending sys.path.

    location = str(site_packages.resolve())
    if location not in sys.path:
        sys.path.append(location)
    import torchaudio

    if str(torchaudio.__version__) != str(_torch.__version__):
        raise RuntimeError(
            f"torchaudio {_torch.__version__} runtime mismatch: {torchaudio.__version__}"
        )
    return str(torchaudio.__version__)


def strict_hrir_from_selected_db(
    selected_db: np.ndarray | torch.Tensor,
    metadata: Mapping[str, np.ndarray | torch.Tensor | int],
) -> torch.Tensor:
    """Reconstruct the frozen strict HRIR with inherited MCA phase/outside bins."""
    values = torch.as_tensor(selected_db, dtype=torch.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("selected_db must have shape [2,direction,frequency]")

    def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(metadata[name], dtype=dtype, device=values.device)

    phase = tensor("mca_selected_phase_rad", torch.float32)
    outside_real = tensor("mca_outside_real", torch.float32)
    outside_imag = tensor("mca_outside_imag", torch.float32)
    selected_indices = tensor("selected_bin_indices_zero_based", torch.int64).reshape(-1)
    outside_indices = tensor("outside_bin_indices_zero_based", torch.int64).reshape(-1)
    single_sided_count = int(metadata["single_sided_frequency_count"])
    hrir_length = int(metadata["hrir_length"])
    if phase.shape != values.shape or outside_real.shape != outside_imag.shape:
        raise ValueError("Strict-HRIR metadata shape mismatch")
    if selected_indices.numel() + outside_indices.numel() != single_sided_count:
        raise ValueError("Strict-HRIR indices do not cover the single-sided spectrum")
    selected = torch.polar(torch.pow(10.0, values / 20.0), phase)
    outside = torch.complex(outside_real, outside_imag)
    spectrum = torch.zeros(
        (2, values.shape[1], single_sided_count),
        dtype=selected.dtype,
        device=values.device,
    )
    spectrum = spectrum.index_copy(-1, outside_indices, outside)
    spectrum = spectrum.index_copy(-1, selected_indices, selected)
    both_sided = torch.cat(
        (spectrum, torch.conj(torch.flip(spectrum[..., 1:-1], dims=(-1,)))), dim=-1
    )
    hrir = torch.fft.ifft(both_sided, dim=-1).real[..., :hrir_length]
    if not bool(torch.all(torch.isfinite(hrir))):
        raise FloatingPointError("Strict HRIR is non-finite")
    return hrir


def absolute_distribution_summary(
    values: np.ndarray, *, gate: bool = False
) -> dict[str, float]:
    """Return the preregistered absolute-value distribution summaries."""
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if absolute.size == 0 or not np.all(np.isfinite(absolute)):
        raise ValueError("Distribution input must be nonempty and finite")
    summary = {
        "MeanAbs": float(np.mean(absolute)),
        "MedianAbs": float(np.median(absolute)),
        "P95Abs": float(np.quantile(absolute, 0.95, method="linear")),
        "P99Abs": float(np.quantile(absolute, 0.99, method="linear")),
        "MaxAbs": float(np.max(absolute)),
    }
    if gate:
        summary["FractionAbsBelow0p01"] = float(np.mean(absolute < 0.01))
        summary["FractionAbsAtLeast0p45"] = float(np.mean(absolute >= 0.45))
    return summary


def dominant_notch_locations_hz(
    spectra_db: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    minimum_hz: float = 4000.0,
    maximum_hz: float = 18000.0,
    smoothing_window_bins: int = 11,
    smoothing_degree: int = 3,
    minimum_prominence_db: float = 1.0,
    minimum_separation_hz: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect one most-prominent DTF notch per ear and direction."""
    spectra = np.asarray(spectra_db, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    weights = normalized_direction_weights(direction_weights)
    if spectra.ndim != 3 or spectra.shape[0] != 2:
        raise ValueError("spectra_db must have shape [2,direction,frequency]")
    if spectra.shape[1] != weights.size or spectra.shape[2] != frequency.size:
        raise ValueError("Notch inputs have incompatible shapes")
    if smoothing_window_bins % 2 != 1 or smoothing_degree >= smoothing_window_bins:
        raise ValueError("Savitzky-Golay window must be odd and exceed its degree")
    spacing = np.diff(frequency)
    if np.any(spacing <= 0.0) or not np.allclose(spacing, spacing[0], rtol=1e-5):
        raise ValueError("Dominant-notch detector requires a uniform frequency grid")
    search = (frequency >= minimum_hz) & (frequency <= maximum_hz)
    search_indices = np.flatnonzero(search)
    if search_indices.size < smoothing_window_bins:
        raise ValueError("Notch search range is too short")
    minimum_distance_bins = max(1, int(round(minimum_separation_hz / spacing[0])))
    dtf = spectra - np.sum(spectra * weights[None, :, None], axis=1)[:, None, :]
    smoothed = savgol_filter(
        dtf,
        window_length=smoothing_window_bins,
        polyorder=smoothing_degree,
        axis=-1,
        mode="interp",
    )
    locations = np.full(spectra.shape[:2], np.nan, dtype=np.float64)
    prominences = np.full(spectra.shape[:2], np.nan, dtype=np.float64)
    for ear in range(2):
        for direction in range(spectra.shape[1]):
            peaks, properties = find_peaks(
                -smoothed[ear, direction, search],
                prominence=minimum_prominence_db,
                distance=minimum_distance_bins,
            )
            if peaks.size == 0:
                continue
            prominence = properties["prominences"]
            best = int(np.flatnonzero(prominence == np.max(prominence))[0])
            locations[ear, direction] = frequency[search_indices[peaks[best]]]
            prominences[ear, direction] = prominence[best]
    return locations, prominences


def dominant_notch_location_metrics(
    predicted_db: np.ndarray,
    reference_db: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    matching_tolerance_hz: float = 1500.0,
) -> dict[str, float]:
    """Score dominant-notch matching with an explicit unmatched penalty."""
    predicted, _ = dominant_notch_locations_hz(
        predicted_db, direction_weights, frequency_hz
    )
    reference, _ = dominant_notch_locations_hz(
        reference_db, direction_weights, frequency_hz
    )
    base_weights = normalized_direction_weights(direction_weights)
    weights = np.broadcast_to(base_weights[None, :] / 2.0, reference.shape)
    reference_valid = np.isfinite(reference)
    predicted_valid = np.isfinite(predicted)
    if not np.any(reference_valid):
        raise ValueError("Reference contains no qualified dominant notches")
    difference = np.abs(predicted - reference)
    matched = reference_valid & predicted_valid & (difference <= matching_tolerance_hz)
    penalized = np.full(reference.shape, matching_tolerance_hz, dtype=np.float64)
    penalized[reference_valid & predicted_valid] = np.minimum(
        difference[reference_valid & predicted_valid], matching_tolerance_hz
    )
    reference_weights = weights[reference_valid]
    reference_weights = reference_weights / reference_weights.sum()
    matched_weights = weights[matched]
    if not np.any(matched):
        matched_mae = matching_tolerance_hz
    else:
        matched_weights = matched_weights / matched_weights.sum()
        matched_mae = float(np.sum(difference[matched] * matched_weights))
    reference_missing = reference_valid & ~matched
    no_reference = ~reference_valid
    spurious = no_reference & predicted_valid
    spurious_rate = (
        float(np.sum(weights[spurious]) / np.sum(weights[no_reference]))
        if np.any(no_reference)
        else 0.0
    )
    return {
        "DominantNotchPenalizedMAE_Hz": float(
            np.sum(penalized[reference_valid] * reference_weights)
        ),
        "DominantNotchMatchedMAE_Hz": matched_mae,
        "DominantNotchMissRate": float(
            np.sum(weights[reference_missing]) / np.sum(weights[reference_valid])
        ),
        "DominantNotchSpuriousRate": spurious_rate,
        "ReferenceNotchFraction": float(np.sum(weights[reference_valid])),
    }
