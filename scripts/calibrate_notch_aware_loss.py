"""Calibrate a validation-only multi-scale notch-depth objective.

This script reads the fixed v3.2 epoch-39 validation predictions and never
touches the consumed test split. It prints candidate objective scales without
writing model or result artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "processed" / "sonicom_residual_q26_v1"
PREDICTION_ROOT = (
    ROOT
    / "artifacts"
    / "reconstruction"
    / "sonicom_q26_validation_mlp_cnn_v32_seed20260809_e40"
)
MINIMUM_FREQUENCY_HZ = 4_000.0
MAXIMUM_FREQUENCY_HZ = 18_000.0
TARGET_STD_DB = 4.954314859581426
REFERENCE_TOTAL = 0.661115484


def softplus(value: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    return np.maximum(value, 0.0) + np.log1p(np.exp(-np.abs(value)))


def notch_depth_map(
    magnitude_db: np.ndarray,
    frequency_hz: np.ndarray,
    radius_bins: int,
    threshold_db: float,
    temperature_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a differentiable local-notch depth map and retained centers."""
    if magnitude_db.ndim != 3:
        raise ValueError("magnitude_db must have shape [ear,direction,frequency]")
    if radius_bins < 1 or 2 * radius_bins >= magnitude_db.shape[-1]:
        raise ValueError("radius_bins is invalid for the frequency axis")
    shoulder_db = 0.5 * (
        magnitude_db[..., : -2 * radius_bins]
        + magnitude_db[..., 2 * radius_bins :]
    )
    center_db = magnitude_db[..., radius_bins:-radius_bins]
    raw_depth_db = shoulder_db - center_db
    depth_db = temperature_db * softplus(
        (raw_depth_db - threshold_db) / temperature_db
    )
    center_frequency_hz = frequency_hz[radius_bins:-radius_bins]
    mask = (
        (center_frequency_hz >= MINIMUM_FREQUENCY_HZ)
        & (center_frequency_hz <= MAXIMUM_FREQUENCY_HZ)
    )
    return depth_db[..., mask], center_frequency_hz[mask]


def direction_weighted_mean(
    values: np.ndarray,
    direction_weights: np.ndarray,
) -> float:
    normalized = direction_weights / np.sum(direction_weights)
    return float(
        np.mean(np.sum(values * normalized[None, :, None], axis=1))
    )


def subject_metric(
    predicted_db: np.ndarray,
    reference_db: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
    radii_bins: tuple[int, ...],
    threshold_db: float,
    temperature_db: float,
) -> tuple[float, float, float]:
    errors: list[float] = []
    reference_depths: list[float] = []
    active_fractions: list[float] = []
    for radius_bins in radii_bins:
        predicted_depth, centers = notch_depth_map(
            predicted_db,
            frequency_hz,
            radius_bins,
            threshold_db,
            temperature_db,
        )
        reference_depth, reference_centers = notch_depth_map(
            reference_db,
            frequency_hz,
            radius_bins,
            threshold_db,
            temperature_db,
        )
        if not np.array_equal(centers, reference_centers):
            raise RuntimeError("Notch center grids do not match")
        errors.append(
            direction_weighted_mean(
                np.abs(predicted_depth - reference_depth), direction_weights
            )
        )
        reference_depths.append(
            direction_weighted_mean(reference_depth, direction_weights)
        )
        active_fractions.append(float(np.mean(reference_depth >= 0.25)))
    return (
        float(np.mean(errors)),
        float(np.mean(reference_depths)),
        float(np.mean(active_fractions)),
    )


def main() -> None:
    report = json.loads(
        (PREDICTION_ROOT / "inference_report.json").read_text(encoding="utf-8")
    )
    if report.get("split") not in {"val", "validation"}:
        raise RuntimeError("Calibration prediction root is not validation")
    subject_labels = [
        str(value["subject_label"]) for value in report["subjects"]
    ]
    configurations = [
        ((4, 8, 16), 0.5, 0.25),
        ((4, 8, 16), 1.0, 0.25),
        ((4, 8, 16), 1.0, 0.5),
        ((8, 16, 24), 1.0, 0.5),
    ]
    summaries: list[dict[str, object]] = []
    for radii_bins, threshold_db, temperature_db in configurations:
        subject_values: list[float] = []
        reference_depths: list[float] = []
        active_fractions: list[float] = []
        for subject_label in subject_labels:
            source_path = DATASET_ROOT / "subjects" / subject_label / "q26.h5"
            prediction_path = (
                PREDICTION_ROOT / "subjects" / subject_label / "prediction.h5"
            )
            with h5py.File(source_path, "r") as source_handle:
                split = source_handle.attrs["split"]
                if isinstance(split, bytes):
                    split = split.decode("utf-8")
                if str(split) not in {"val", "validation"}:
                    raise RuntimeError(f"Unexpected split for {subject_label}: {split}")
                interpolation_mask = np.squeeze(
                    np.asarray(
                        source_handle["interpolation_evaluation_mask"],
                        dtype=np.uint8,
                    )
                ).astype(bool)
                mca_db = np.asarray(
                    source_handle["mca_logmag_db"][:, interpolation_mask, :],
                    dtype=np.float64,
                )
                target_db = np.asarray(
                    source_handle["target_residual_db"][:, interpolation_mask, :],
                    dtype=np.float64,
                )
                direction_weights = np.asarray(
                    source_handle["direction_features"][interpolation_mask, 5],
                    dtype=np.float64,
                )
                frequency_hz = np.squeeze(
                    np.asarray(source_handle["frequency_hz"], dtype=np.float64)
                )
            with h5py.File(prediction_path, "r") as prediction_handle:
                prediction_db = np.asarray(
                    prediction_handle["predicted_residual_db"][
                        :, interpolation_mask, :
                    ],
                    dtype=np.float64,
                )
            metric, reference_depth, active_fraction = subject_metric(
                mca_db + prediction_db,
                mca_db + target_db,
                direction_weights,
                frequency_hz,
                radii_bins,
                threshold_db,
                temperature_db,
            )
            subject_values.append(metric)
            reference_depths.append(reference_depth)
            active_fractions.append(active_fraction)

        mean_metric = float(np.mean(subject_values))
        target_contribution = 0.03
        calibrated_weight = target_contribution * TARGET_STD_DB / mean_metric
        summaries.append(
            {
                "radii_bins": list(radii_bins),
                "radii_hz": [float(value * 43.06640625) for value in radii_bins],
                "threshold_db": threshold_db,
                "temperature_db": temperature_db,
                "subject_count": len(subject_values),
                "notch_depth_mae_db": mean_metric,
                "reference_notch_depth_mean_db": float(
                    np.mean(reference_depths)
                ),
                "reference_depth_ge_0p25_fraction": float(
                    np.mean(active_fractions)
                ),
                "weight_for_0p03_total_contribution": calibrated_weight,
                "fraction_of_new_initial_total_percent": float(
                    100.0 * target_contribution / (REFERENCE_TOTAL + target_contribution)
                ),
                "minimum_frequency_hz": MINIMUM_FREQUENCY_HZ,
                "maximum_frequency_hz": MAXIMUM_FREQUENCY_HZ,
                "test_subject_count_read": 0,
            }
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
