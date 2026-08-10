"""Summarize the validation-only v3.4c notch-aware ablation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "sonicom_mlp_cnn_q26_v34c_notch_aware_e10_strict_validation"
)
DATASET_ROOT = ROOT / "data" / "processed" / "sonicom_residual_q26_v1"
BASELINE_ROOT = (
    ROOT
    / "artifacts"
    / "reconstruction"
    / "sonicom_q26_validation_mlp_cnn_v32_seed20260809_e40"
)
CANDIDATE_ROOT = (
    ROOT
    / "artifacts"
    / "reconstruction"
    / "sonicom_q26_validation_mlp_cnn_v34c_notch_aware_e10"
)
SOURCE_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_seed20260809_e40"
    / "best.pt"
)
CANDIDATE_TRAINING_ROOT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v34c_notch_aware_e10"
)
CANDIDATE_CHECKPOINT = CANDIDATE_TRAINING_ROOT / "best.pt"
MINIMUM_FREQUENCY_HZ = 4_000.0
MAXIMUM_FREQUENCY_HZ = 18_000.0
RADII_BINS = (4, 8, 16)
DEPTH_THRESHOLD_DB = 1.0
SOFTPLUS_TEMPERATURE_DB = 0.5


METRICS = {
    "FullSphereERB": (
        "MLPCNNv3FullSphereERB_dB",
        "MLPCNNv31FullSphereERB_dB",
    ),
    "Contralateral25ERB": (
        "MLPCNNv3Contralateral25ERB_dB",
        "MLPCNNv31Contralateral25ERB_dB",
    ),
    "ContralateralHighFrequency": (
        "MLPCNNv3ContralateralHighFrequency_dB",
        "MLPCNNv31ContralateralHighFrequency_dB",
    ),
    "HorizontalILDMAE": (
        "MLPCNNv3HorizontalILDMAE_dB",
        "MLPCNNv31HorizontalILDMAE_dB",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def paired_row(
    metric: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    difference = candidate - baseline
    interval = stats.t.interval(
        0.95,
        len(difference) - 1,
        loc=np.mean(difference),
        scale=stats.sem(difference),
    )
    difference_std = np.std(difference, ddof=1)
    return {
        "metric": metric,
        "subject_count": len(difference),
        "baseline_mean": float(np.mean(baseline)),
        "candidate_mean": float(np.mean(candidate)),
        "mean_candidate_minus_baseline": float(np.mean(difference)),
        "improvement_vs_baseline_percent": float(
            100.0 * (np.mean(baseline) - np.mean(candidate)) / np.mean(baseline)
        ),
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
        "paired_t_p_value": float(stats.ttest_rel(candidate, baseline).pvalue),
        "wilcoxon_p_value": float(stats.wilcoxon(candidate, baseline).pvalue),
        "cohen_dz": float(np.mean(difference) / difference_std),
        "candidate_better_subject_count": int(np.sum(difference < 0.0)),
    }


def softplus(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, 0.0) + np.log1p(np.exp(-np.abs(value)))


def notch_depth_map(
    magnitude_db: np.ndarray,
    frequency_hz: np.ndarray,
    radius_bins: int,
) -> np.ndarray:
    shoulder_db = 0.5 * (
        magnitude_db[..., : -2 * radius_bins]
        + magnitude_db[..., 2 * radius_bins :]
    )
    center_db = magnitude_db[..., radius_bins:-radius_bins]
    raw_depth_db = shoulder_db - center_db
    depth_db = SOFTPLUS_TEMPERATURE_DB * softplus(
        (raw_depth_db - DEPTH_THRESHOLD_DB) / SOFTPLUS_TEMPERATURE_DB
    )
    centers = frequency_hz[radius_bins:-radius_bins]
    mask = (
        (centers >= MINIMUM_FREQUENCY_HZ)
        & (centers <= MAXIMUM_FREQUENCY_HZ)
    )
    return depth_db[..., mask]


def notch_metrics(
    prediction_db: np.ndarray,
    reference_db: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[float, np.ndarray]:
    normalized = direction_weights / np.sum(direction_weights)
    weights = normalized[None, :, None]
    by_radius: list[float] = []
    for radius_bins in RADII_BINS:
        prediction_depth = notch_depth_map(
            prediction_db, frequency_hz, radius_bins
        )
        reference_depth = notch_depth_map(
            reference_db, frequency_hz, radius_bins
        )
        error = np.abs(prediction_depth - reference_depth)
        by_radius.append(float(np.mean(np.sum(error * weights, axis=1))))
    values = np.asarray(by_radius, dtype=np.float64)
    return float(np.mean(values)), values


def parameter_group_diagnostics(
    source_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, object]:
    squared_delta = 0.0
    squared_source = 0.0
    maximum_delta = 0.0
    tensor_count = 0
    parameter_count = 0
    for key, source_value in source_state.items():
        if not key.startswith(prefix) or not torch.is_floating_point(source_value):
            continue
        candidate_value = candidate_state[key]
        delta = candidate_value.double() - source_value.double()
        squared_delta += float(torch.sum(delta * delta))
        squared_source += float(torch.sum(source_value.double() ** 2))
        maximum_delta = max(maximum_delta, float(torch.max(torch.abs(delta))))
        tensor_count += 1
        parameter_count += source_value.numel()
    l2_delta = squared_delta**0.5
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "maximum_absolute_parameter_delta": maximum_delta,
        "l2_parameter_delta": l2_delta,
        "relative_l2_parameter_delta": l2_delta / squared_source**0.5,
        "bitwise_unchanged": squared_delta == 0.0,
    }


def main() -> None:
    summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    if summary["split"] not in {"val", "validation"}:
        raise RuntimeError("Strict result is not validation")
    if summary["test_subject_count_read"] != 0 or summary["subject_count"] != 44:
        raise RuntimeError("Strict result violated the validation-only boundary")
    per_subject = pd.read_csv(RESULT_ROOT / "per_subject_metrics.csv")
    subject_labels = per_subject["SubjectLabel"].astype(str).tolist()

    strict_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for metric, (baseline_column, candidate_column) in METRICS.items():
        baseline = per_subject[baseline_column].to_numpy(dtype=np.float64)
        candidate = per_subject[candidate_column].to_numpy(dtype=np.float64)
        row = paired_row(metric, baseline, candidate)
        strict_rows.append(row)
        comparison_rows.append(
            {
                "metric": metric,
                "v32_epoch39_mean_db": row["baseline_mean"],
                "v34c_epoch7_mean_db": row["candidate_mean"],
                "improvement_vs_v32_percent": row[
                    "improvement_vs_baseline_percent"
                ],
                "v34c_better_subject_count": row[
                    "candidate_better_subject_count"
                ],
                "subject_count": 44,
            }
        )

    notch_rows: list[dict[str, object]] = []
    baseline_by_radius: list[np.ndarray] = []
    candidate_by_radius: list[np.ndarray] = []
    for subject_label in subject_labels:
        source_path = DATASET_ROOT / "subjects" / subject_label / "q26.h5"
        baseline_path = BASELINE_ROOT / "subjects" / subject_label / "prediction.h5"
        candidate_path = CANDIDATE_ROOT / "subjects" / subject_label / "prediction.h5"
        with h5py.File(source_path, "r") as source_handle:
            interpolation_mask = np.squeeze(
                np.asarray(
                    source_handle["interpolation_evaluation_mask"], dtype=np.uint8
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
        with h5py.File(baseline_path, "r") as handle:
            baseline_residual = np.asarray(
                handle["predicted_residual_db"][:, interpolation_mask, :],
                dtype=np.float64,
            )
        with h5py.File(candidate_path, "r") as handle:
            candidate_residual = np.asarray(
                handle["predicted_residual_db"][:, interpolation_mask, :],
                dtype=np.float64,
            )
        reference_db = mca_db + target_db
        baseline_metric, baseline_radius = notch_metrics(
            mca_db + baseline_residual,
            reference_db,
            direction_weights,
            frequency_hz,
        )
        candidate_metric, candidate_radius = notch_metrics(
            mca_db + candidate_residual,
            reference_db,
            direction_weights,
            frequency_hz,
        )
        baseline_by_radius.append(baseline_radius)
        candidate_by_radius.append(candidate_radius)
        notch_rows.append(
            {
                "subject_label": subject_label,
                "interpolation_direction_count": int(np.sum(interpolation_mask)),
                "v32_notch_depth_mae_db": baseline_metric,
                "v34c_notch_depth_mae_db": candidate_metric,
                "improvement_percent": 100.0
                * (baseline_metric - candidate_metric)
                / baseline_metric,
            }
        )

    notch_frame = pd.DataFrame(notch_rows)
    notch_statistics = paired_row(
        "MultiScaleNotchDepthMAE_dB",
        notch_frame["v32_notch_depth_mae_db"].to_numpy(),
        notch_frame["v34c_notch_depth_mae_db"].to_numpy(),
    )
    baseline_radius_matrix = np.stack(baseline_by_radius)
    candidate_radius_matrix = np.stack(candidate_by_radius)
    radius_rows: list[dict[str, object]] = []
    bin_spacing_hz = 43.06640625
    for index, radius_bins in enumerate(RADII_BINS):
        baseline = baseline_radius_matrix[:, index]
        candidate = candidate_radius_matrix[:, index]
        radius_rows.append(
            {
                "radius_bins": radius_bins,
                "radius_hz": radius_bins * bin_spacing_hz,
                "v32_notch_depth_mae_db": float(np.mean(baseline)),
                "v34c_notch_depth_mae_db": float(np.mean(candidate)),
                "improvement_percent": float(
                    100.0 * (np.mean(baseline) - np.mean(candidate)) / np.mean(baseline)
                ),
                "v34c_better_subject_count": int(np.sum(candidate < baseline)),
                "subject_count": 44,
            }
        )

    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    candidate = torch.load(
        CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    source_state = source["model_state"]
    candidate_state = candidate["model_state"]
    training_report = json.loads(
        (CANDIDATE_TRAINING_ROOT / "training_report.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = {
        "schema_version": "1.0",
        "split": "val",
        "test_subject_count_read": 0,
        "subject_count": 44,
        "notch_depth": {
            "definition": {
                "minimum_frequency_hz": MINIMUM_FREQUENCY_HZ,
                "maximum_frequency_hz": MAXIMUM_FREQUENCY_HZ,
                "radii_bins": list(RADII_BINS),
                "depth_threshold_db": DEPTH_THRESHOLD_DB,
                "softplus_temperature_db": SOFTPLUS_TEMPERATURE_DB,
                "direction_subset": "all interpolation directions",
                "spatial_weighting": "normalized solid-angle weights per subject",
            },
            "paired_statistics": notch_statistics,
            "by_radius": radius_rows,
        },
        "checkpoint": {
            "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
            "candidate_checkpoint_sha256": sha256(CANDIDATE_CHECKPOINT),
            "candidate_checkpoint_epoch": int(candidate["epoch"]),
            "mlp": parameter_group_diagnostics(
                source_state, candidate_state, "mlp."
            ),
            "local_cnn": parameter_group_diagnostics(
                source_state, candidate_state, "cnn."
            ),
        },
        "training": training_report,
    }
    write_csv(RESULT_ROOT / "comparison_vs_v32_epoch39.csv", comparison_rows)
    write_csv(RESULT_ROOT / "paired_statistics.csv", strict_rows)
    write_csv(RESULT_ROOT / "notch_depth_per_subject.csv", notch_rows)
    write_csv(RESULT_ROOT / "notch_depth_paired_statistics.csv", [notch_statistics])
    write_csv(RESULT_ROOT / "notch_depth_by_radius.csv", radius_rows)
    (RESULT_ROOT / "notch_depth_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
