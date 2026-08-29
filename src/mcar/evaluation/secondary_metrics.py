"""Result-blind secondary metrics for FiLM-family validation studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from mcar.losses import (
    high_frequency_spectral_difference_mae,
    multi_scale_notch_depth_mae,
    spectral_band_ild_smooth_l1_and_mae,
)
from mcar.training.train_mlp_v2 import (
    make_erb_center_frequencies_hz,
    make_erb_weights,
)


DISTANCE_BIN_LABELS = ("0_10", "10_20", "20_30", "30_180")
DISTANCE_BIN_EDGES_DEG = (0.0, 10.0, 20.0, 30.0, 180.0)


def _spectral_arrays(
    predicted_db: np.ndarray, reference_db: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_db, dtype=np.float64)
    reference = np.asarray(reference_db, dtype=np.float64)
    if predicted.ndim != 3 or predicted.shape[0] != 2:
        raise ValueError("Spectra must have shape [2,direction,frequency]")
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference spectra must share shape")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(reference)):
        raise FloatingPointError("Spectra must be finite")
    return predicted, reference


def normalized_direction_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Direction weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Direction weights must have positive sum")
    return values / total


def full_sphere_lsd(
    predicted_db: np.ndarray,
    reference_db: np.ndarray,
    interpolation_mask: np.ndarray,
    direction_weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return subject LSD plus unmasked direction-mean and ear maps."""
    predicted, reference = _spectral_arrays(predicted_db, reference_db)
    mask = np.asarray(interpolation_mask, dtype=bool).reshape(-1)
    weights = np.asarray(direction_weights, dtype=np.float64).reshape(-1)
    if mask.shape != (predicted.shape[1],) or weights.shape != mask.shape:
        raise ValueError("Mask/weights must match the direction axis")
    ear_direction_lsd = np.sqrt(np.mean((predicted - reference) ** 2, axis=2))
    direction_lsd = ear_direction_lsd.mean(axis=0)
    value = float(
        np.sum(direction_lsd[mask] * normalized_direction_weights(weights[mask]))
    )
    return value, direction_lsd, ear_direction_lsd


def minimum_q26_distance_degrees(
    xyz: np.ndarray, q26_mask: np.ndarray
) -> np.ndarray:
    """Return minimum great-circle distance from every direction to Q26."""
    vectors = np.asarray(xyz, dtype=np.float64)
    mask = np.asarray(q26_mask, dtype=bool).reshape(-1)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or mask.shape != (vectors.shape[0],):
        raise ValueError("xyz must be [direction,3] and mask must match")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(vectors)):
        raise ValueError("Direction vectors must be finite and non-zero")
    if np.count_nonzero(mask) != 26:
        raise ValueError("Exactly 26 Q26 directions are required")
    unit = vectors / norms[:, None]
    cosine = np.clip(unit @ unit[mask].T, -1.0, 1.0)
    return np.rad2deg(np.arccos(np.max(cosine, axis=1)))


def distance_binned_lsd(
    direction_lsd_db: np.ndarray,
    distance_deg: np.ndarray,
    interpolation_mask: np.ndarray,
    direction_weights: np.ndarray,
) -> dict[str, float]:
    """Aggregate direction LSD in the four preregistered Q26-distance bins."""
    lsd = np.asarray(direction_lsd_db, dtype=np.float64).reshape(-1)
    distance = np.asarray(distance_deg, dtype=np.float64).reshape(-1)
    interpolation = np.asarray(interpolation_mask, dtype=bool).reshape(-1)
    weights = np.asarray(direction_weights, dtype=np.float64).reshape(-1)
    if not (lsd.shape == distance.shape == interpolation.shape == weights.shape):
        raise ValueError("Direction arrays must share shape")
    output: dict[str, float] = {}
    assigned = np.zeros(lsd.shape, dtype=bool)
    for index, label in enumerate(DISTANCE_BIN_LABELS):
        lower = DISTANCE_BIN_EDGES_DEG[index]
        upper = DISTANCE_BIN_EDGES_DEG[index + 1]
        current = interpolation & (distance >= lower)
        current &= distance <= upper + 1e-10 if index == 3 else distance < upper
        if not np.any(current):
            raise ValueError(f"Distance bin {label} is empty")
        assigned |= current
        output[label] = float(
            np.sum(lsd[current] * normalized_direction_weights(weights[current]))
        )
    if not np.array_equal(assigned, interpolation):
        raise ValueError("Distance bins do not partition the interpolation mask")
    return output


def spectral_shape_metrics(
    predicted_db: np.ndarray,
    reference_db: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
) -> dict[str, float]:
    """Reuse the frozen repository HF-difference and notch-depth definitions."""
    predicted, reference = _spectral_arrays(predicted_db, reference_db)
    weights = np.asarray(direction_weights, dtype=np.float32).reshape(-1)
    frequency = np.asarray(frequency_hz, dtype=np.float32).reshape(-1)
    with torch.no_grad():
        prediction_tensor = torch.from_numpy(predicted.astype(np.float32))
        reference_tensor = torch.from_numpy(reference.astype(np.float32))
        weight_tensor = torch.from_numpy(weights)
        frequency_tensor = torch.from_numpy(frequency)
        first, second = high_frequency_spectral_difference_mae(
            prediction_tensor,
            reference_tensor,
            weight_tensor,
            frequency_tensor,
            minimum_frequency_hz=4000.0,
        )
        notch = multi_scale_notch_depth_mae(
            prediction_tensor,
            reference_tensor,
            weight_tensor,
            frequency_tensor,
            minimum_frequency_hz=4000.0,
            maximum_frequency_hz=18000.0,
            radii_bins=(4, 8, 16),
            depth_threshold_db=1.0,
            softplus_temperature_db=0.5,
        )
    return {
        "HFFirstDifferenceMAE": float(first.item()),
        "HFSecondDifferenceMAE": float(second.item()),
        "MultiScaleNotchDepthMAE": float(notch.item()),
    }


def spectral_band_ild_profile(
    predicted_db: np.ndarray,
    reference_db: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return registered 200--18000 Hz ERB-band ILD MAE profile and mean."""
    predicted, reference = _spectral_arrays(predicted_db, reference_db)
    weights = np.asarray(direction_weights, dtype=np.float32).reshape(-1)
    frequency = np.asarray(frequency_hz, dtype=np.float32).reshape(-1)
    raw_band_weights = make_erb_weights(frequency)
    centers = make_erb_center_frequencies_hz(raw_band_weights.shape[0])
    band_mask = (centers >= 200.0) & (centers <= 18000.0)
    log_weights = torch.log(
        torch.clamp(torch.from_numpy(raw_band_weights), min=1e-12)
    ).view(1, 1, raw_band_weights.shape[0], frequency.size)
    prediction_tensor = torch.from_numpy(predicted.astype(np.float32))
    reference_tensor = torch.from_numpy(reference.astype(np.float32))
    scale = np.log(10.0) / 10.0
    with torch.no_grad():
        def band_energy_db(values: torch.Tensor) -> torch.Tensor:
            return (1.0 / scale) * torch.logsumexp(
                scale * values.unsqueeze(-2) + log_weights, dim=-1
            )

        predicted_band = band_energy_db(prediction_tensor)[..., band_mask]
        reference_band = band_energy_db(reference_tensor)[..., band_mask]
        absolute_error = torch.abs(
            (predicted_band[0] - predicted_band[1])
            - (reference_band[0] - reference_band[1])
        )
        normalized = torch.from_numpy(normalized_direction_weights(weights).astype(np.float32))
        profile = torch.sum(absolute_error * normalized[:, None], dim=0)
        _, registered_mean = spectral_band_ild_smooth_l1_and_mae(
            prediction_tensor,
            reference_tensor,
            torch.from_numpy(weights),
            log_weights,
            torch.from_numpy(centers),
            minimum_band_center_hz=200.0,
            maximum_band_center_hz=18000.0,
            beta_db=0.5,
        )
    profile_np = profile.cpu().numpy().astype(np.float64)
    mean_value = float(registered_mean.item())
    if not np.isclose(profile_np.mean(), mean_value, rtol=2e-6, atol=2e-6):
        raise AssertionError("Band-profile mean disagrees with registered loss")
    return centers[band_mask].astype(np.float64), profile_np, mean_value


def bootstrap_mean_interval(
    values: Sequence[float], *, replicates: int = 10000, seed: int = 20260829
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Bootstrap input must be nonempty and finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(replicates, data.size))
    means = data[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return float(data.mean()), float(lower), float(upper)


def paired_tail_statistics(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    tie_tolerance: float = 1e-12,
    replicates: int = 10000,
    seed: int = 20260829,
) -> dict[str, float | int | str]:
    """Return preregistered subject-paired central and tail statistics."""
    if set(candidate) != set(baseline) or len(candidate) != 44:
        raise ValueError("Paired statistics require identical sets of 44 subjects")
    subject_ids = sorted(candidate)
    differences = np.asarray(
        [candidate[item] - baseline[item] for item in subject_ids], dtype=np.float64
    )
    if not np.all(np.isfinite(differences)):
        raise FloatingPointError("Paired differences must be finite")
    mean, lower, upper = bootstrap_mean_interval(
        differences, replicates=replicates, seed=seed
    )
    ordering = np.argsort(differences)
    maximum_index = int(ordering[-1])
    ties = np.abs(differences) <= tie_tolerance
    return {
        "SubjectCount": int(differences.size),
        "MeanDifference": mean,
        "Bootstrap95Lower": lower,
        "Bootstrap95Upper": upper,
        "MedianDifference": float(np.median(differences)),
        "P90Difference": float(np.quantile(differences, 0.90, method="linear")),
        "P95Difference": float(np.quantile(differences, 0.95, method="linear")),
        "WorstDecileMean": float(np.mean(differences[ordering[-5:]])),
        "MaximumDegradation": float(differences[maximum_index]),
        "MaximumSubject": subject_ids[maximum_index],
        "Wins": int(np.count_nonzero(differences < -tie_tolerance)),
        "Ties": int(np.count_nonzero(ties)),
        "Losses": int(np.count_nonzero(differences > tie_tolerance)),
        "WinRate": float(np.count_nonzero(differences < -tie_tolerance) / differences.size),
    }
