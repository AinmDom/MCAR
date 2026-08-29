"""Synthetic qualification tests for preregistered secondary metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mcar.evaluation.secondary_metrics import (
    bootstrap_mean_interval,
    distance_binned_lsd,
    full_sphere_lsd,
    minimum_q26_distance_degrees,
    paired_tail_statistics,
    spectral_band_ild_profile,
    spectral_shape_metrics,
)


def test_full_sphere_lsd_is_ear_mean_then_weighted_direction_mean() -> None:
    reference = np.zeros((2, 3, 4), dtype=np.float32)
    predicted = reference.copy()
    predicted[:, 0, :] = 1.0
    predicted[:, 1, :] = 3.0
    predicted[:, 2, :] = 100.0
    value, direction, ear_direction = full_sphere_lsd(
        predicted,
        reference,
        np.array([True, True, False]),
        np.array([0.75, 0.25, 99.0]),
    )
    np.testing.assert_allclose(direction, [1.0, 3.0, 100.0])
    np.testing.assert_allclose(ear_direction, [[1.0, 3.0, 100.0]] * 2)
    assert value == pytest.approx(1.5)


def test_q26_distance_and_bins_form_exact_partition() -> None:
    angles = np.deg2rad(np.arange(30.0))
    xyz = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(30)))
    q26 = np.zeros(30, dtype=bool)
    q26[:26] = True
    distance = minimum_q26_distance_degrees(xyz, q26)
    np.testing.assert_allclose(distance[:26], 0.0, atol=1e-6)

    lsd = np.array([1.0, 2.0, 3.0, 4.0])
    values = distance_binned_lsd(
        lsd,
        np.array([0.0, 10.0, 20.0, 30.0]),
        np.ones(4, dtype=bool),
        np.ones(4),
    )
    assert values == {
        "0_10": 1.0,
        "10_20": 2.0,
        "20_30": 3.0,
        "30_180": 4.0,
    }


def test_spectral_shape_metrics_match_known_polynomial_errors() -> None:
    frequency = np.linspace(1000.0, 20000.0, 50, dtype=np.float32)
    bins = np.arange(50, dtype=np.float32)
    reference = np.zeros((2, 2, 50), dtype=np.float32)
    linear = np.broadcast_to(bins, reference.shape).copy()
    metrics = spectral_shape_metrics(linear, reference, np.array([0.25, 0.75]), frequency)
    assert metrics["HFFirstDifferenceMAE"] == pytest.approx(1.0)
    assert metrics["HFSecondDifferenceMAE"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["MultiScaleNotchDepthMAE"] >= 0.0


def test_band_ild_profile_has_registered_mean_and_zero_identity() -> None:
    frequency = np.linspace(86.0, 19983.0, 463, dtype=np.float32)
    reference = np.zeros((2, 3, 463), dtype=np.float32)
    centers, profile, mean_value = spectral_band_ild_profile(
        reference, reference, np.array([0.2, 0.3, 0.5]), frequency
    )
    assert centers.size == profile.size
    assert np.all((centers >= 200.0) & (centers <= 18000.0))
    np.testing.assert_allclose(profile, 0.0, atol=0.0)
    assert mean_value == pytest.approx(0.0)

    predicted = reference.copy()
    predicted[0, 0, :] = 1.0
    predicted[0, 1, :] = 2.0
    predicted[0, 2, :] = 3.0
    _, profile, mean_value = spectral_band_ild_profile(
        predicted, reference, np.array([0.2, 0.3, 0.5]), frequency
    )
    np.testing.assert_allclose(profile, 2.3, atol=2e-5)
    assert mean_value == pytest.approx(2.3, abs=2e-5)


def test_tail_statistics_use_44_subjects_and_largest_five() -> None:
    candidate = {f"P{i:04d}": float(i) for i in range(44)}
    baseline = {key: 0.0 for key in candidate}
    values = paired_tail_statistics(candidate, baseline, replicates=100, seed=7)
    assert values["MedianDifference"] == pytest.approx(21.5)
    assert values["P90Difference"] == pytest.approx(38.7)
    assert values["P95Difference"] == pytest.approx(40.85)
    assert values["WorstDecileMean"] == pytest.approx(41.0)
    assert values["MaximumDegradation"] == pytest.approx(43.0)
    assert values["MaximumSubject"] == "P0043"
    assert values["Wins"] == 0
    assert values["Ties"] == 1
    assert values["Losses"] == 43


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_mean_interval([1.0, 2.0, 3.0], replicates=200, seed=11)
    second = bootstrap_mean_interval([1.0, 2.0, 3.0], replicates=200, seed=11)
    assert first == second
