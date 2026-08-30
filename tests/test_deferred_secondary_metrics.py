"""Synthetic qualification for deferred FiLM secondary endpoints."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mcar.evaluation.deferred_secondary_metrics import (
    absolute_distribution_summary,
    dominant_notch_location_metrics,
    dominant_notch_locations_hz,
    enable_external_torchaudio,
    strict_hrir_from_selected_db,
)
from mcar.fsp_ae_signal import estimate_itd_seconds
from mcar.predictors import bounded_diagnostics_to_db


def test_bounded_diagnostic_conversion_preserves_delta_units() -> None:
    base = torch.tensor([[[1.0, -1.0]]])
    correction = torch.tensor([[[0.5, -0.25]]])
    gate = torch.tensor([[[0.2, -0.1]]])
    values = bounded_diagnostics_to_db(
        base + correction,
        base,
        correction,
        gate,
        target_mean=-3.0,
        target_std=2.0,
    )
    np.testing.assert_allclose(values.base_residual_db, [[[-1.0, -5.0]]])
    np.testing.assert_allclose(values.applied_correction_db, [[[1.0, -0.5]]])
    np.testing.assert_allclose(
        values.final_residual_db,
        values.base_residual_db + values.applied_correction_db,
    )
    np.testing.assert_allclose(values.gate, gate.numpy())


def test_absolute_summary_uses_linear_quantiles_and_gate_fractions() -> None:
    values = absolute_distribution_summary(
        np.array([-0.5, -0.005, 0.0, 0.25]), gate=True
    )
    assert values["MeanAbs"] == pytest.approx(0.18875)
    assert values["MedianAbs"] == pytest.approx(0.1275)
    assert values["FractionAbsBelow0p01"] == pytest.approx(0.5)
    assert values["FractionAbsAtLeast0p45"] == pytest.approx(0.25)


def synthetic_dtf_notches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.linspace(3000.0, 19000.0, 321)
    weights = np.full(4, 0.25)
    spectra = np.zeros((2, 4, frequency.size), dtype=np.float64)
    for ear in range(2):
        for direction in range(4):
            center = 8000.0 + 500.0 * direction + 250.0 * ear
            spectra[ear, direction] -= 8.0 * np.exp(
                -0.5 * ((frequency - center) / 180.0) ** 2
            )
    return spectra, weights, frequency


def test_dominant_notch_detector_and_matching_are_deterministic() -> None:
    reference, weights, frequency = synthetic_dtf_notches()
    locations, prominence = dominant_notch_locations_hz(
        reference, weights, frequency
    )
    assert np.all(np.isfinite(locations))
    assert np.all(prominence >= 1.0)
    shifted = np.roll(reference, 10, axis=-1)
    shifted[..., :10] = 0.0
    metrics = dominant_notch_location_metrics(
        shifted, reference, weights, frequency
    )
    assert metrics["DominantNotchPenalizedMAE_Hz"] == pytest.approx(500.0, abs=55.0)
    assert metrics["DominantNotchMissRate"] == pytest.approx(0.0)


def test_strict_hrir_reconstruction_matches_direct_spectrum() -> None:
    selected_db = np.zeros((2, 3, 2), dtype=np.float32)
    phase = np.zeros_like(selected_db)
    outside_real = np.full((2, 3, 1), 0.5, dtype=np.float32)
    metadata = {
        "mca_selected_phase_rad": phase,
        "mca_outside_real": outside_real,
        "mca_outside_imag": np.zeros_like(outside_real),
        "selected_bin_indices_zero_based": np.array([1, 2]),
        "outside_bin_indices_zero_based": np.array([0]),
        "single_sided_frequency_count": 3,
        "hrir_length": 4,
    }
    actual = strict_hrir_from_selected_db(selected_db, metadata).numpy()
    one_sided = np.array([0.5, 1.0, 1.0], dtype=np.complex64)
    expected = np.fft.ifft(np.r_[one_sided, np.conj(one_sided[1:-1][::-1])]).real
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-7)


def test_published_itd_estimator_recovers_known_delay() -> None:
    site_packages = os.environ.get("MCAR_TORCHAUDIO_SITE_PACKAGES")
    if not site_packages:
        pytest.skip("exact external torchaudio path was not supplied")
    enable_external_torchaudio(Path(site_packages))
    hrir = torch.zeros((1, 2, 256), dtype=torch.float32)
    hrir[0, 0, 80] = 1.0
    hrir[0, 1, 84] = 1.0
    estimate = estimate_itd_seconds(
        hrir,
        44100.0,
        upsampled_rate_hz=384000.0,
        lowpass_hz=1600.0,
        maximum_itd_seconds=0.001,
        direction_batch_size=1,
    )
    assert abs(abs(float(estimate.item())) - 4.0 / 44100.0) <= 2.0 / 384000.0
