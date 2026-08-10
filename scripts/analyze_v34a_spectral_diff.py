"""Summarize the validation-only v3.4a spectral-difference ablation."""

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
    / "sonicom_mlp_cnn_q26_v34a_spectral_diff_e10_strict_validation"
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
    / "sonicom_q26_validation_mlp_cnn_v34a_spectral_diff_e10"
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
    / "sonicom_mlp_cnn_q26_v34a_spectral_diff_e10"
)
CANDIDATE_CHECKPOINT = CANDIDATE_TRAINING_ROOT / "best.pt"
SPECTRAL_CUTOFF_HZ = 4_000.0


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
    standard_error = stats.sem(difference)
    interval = stats.t.interval(
        0.95,
        len(difference) - 1,
        loc=np.mean(difference),
        scale=standard_error,
    )
    t_result = stats.ttest_rel(candidate, baseline)
    wilcoxon = stats.wilcoxon(candidate, baseline)
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
        "paired_t_p_value": float(t_result.pvalue),
        "wilcoxon_p_value": float(wilcoxon.pvalue),
        "cohen_dz": float(np.mean(difference) / difference_std),
        "candidate_better_subject_count": int(np.sum(difference < 0.0)),
    }


def paired_statistics(per_subject: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, (baseline_column, candidate_column) in METRICS.items():
        rows.append(
            paired_row(
                metric,
                per_subject[baseline_column].to_numpy(dtype=np.float64),
                per_subject[candidate_column].to_numpy(dtype=np.float64),
            )
        )
    return rows


def spectral_difference_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[float, float]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must share [ear,direction,frequency]")
    if direction_weights.shape != (prediction.shape[1],):
        raise ValueError("Direction weights do not match prediction directions")
    if frequency_hz.shape != (prediction.shape[2],):
        raise ValueError("Frequency vector does not match prediction frequencies")
    if np.any(np.diff(frequency_hz) <= 0.0):
        raise ValueError("Frequency vector must be strictly increasing")
    if np.any(direction_weights < 0.0) or np.sum(direction_weights) <= 0.0:
        raise ValueError("Direction weights must be non-negative with positive sum")

    error = prediction.astype(np.float64) - target.astype(np.float64)
    normalized_weights = direction_weights / np.sum(direction_weights)
    first_mask = frequency_hz[:-1] >= SPECTRAL_CUTOFF_HZ
    second_mask = frequency_hz[:-2] >= SPECTRAL_CUTOFF_HZ
    first_error = np.abs(np.diff(error, axis=-1)[..., first_mask])
    second_error = np.abs(np.diff(error, n=2, axis=-1)[..., second_mask])
    weights = normalized_weights[np.newaxis, :, np.newaxis]
    first_mae = np.mean(np.sum(first_error * weights, axis=1))
    second_mae = np.mean(np.sum(second_error * weights, axis=1))
    return float(first_mae), float(second_mae)


def spectral_difference_diagnostics(
    subject_labels: list[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
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
            target = np.asarray(
                source_handle["target_residual_db"][:, interpolation_mask, :],
                dtype=np.float32,
            )
            direction_features = np.asarray(
                source_handle["direction_features"][interpolation_mask, :],
                dtype=np.float64,
            )
            direction_weights = direction_features[:, 5]
            frequency_hz = np.squeeze(
                np.asarray(source_handle["frequency_hz"], dtype=np.float64)
            )
        with h5py.File(baseline_path, "r") as baseline_handle:
            baseline = np.asarray(
                baseline_handle["predicted_residual_db"][:, interpolation_mask, :],
                dtype=np.float32,
            )
        with h5py.File(candidate_path, "r") as candidate_handle:
            candidate = np.asarray(
                candidate_handle["predicted_residual_db"][:, interpolation_mask, :],
                dtype=np.float32,
            )

        baseline_d1, baseline_d2 = spectral_difference_metrics(
            baseline, target, direction_weights, frequency_hz
        )
        candidate_d1, candidate_d2 = spectral_difference_metrics(
            candidate, target, direction_weights, frequency_hz
        )
        rows.append(
            {
                "subject_label": subject_label,
                "interpolation_direction_count": int(np.sum(interpolation_mask)),
                "v32_first_difference_mae_db_per_bin": baseline_d1,
                "v34a_first_difference_mae_db_per_bin": candidate_d1,
                "first_difference_improvement_percent": 100.0
                * (baseline_d1 - candidate_d1)
                / baseline_d1,
                "v32_second_difference_mae_db_per_bin2": baseline_d2,
                "v34a_second_difference_mae_db_per_bin2": candidate_d2,
                "second_difference_improvement_percent": 100.0
                * (baseline_d2 - candidate_d2)
                / baseline_d2,
            }
        )

    frame = pd.DataFrame(rows)
    first_statistics = paired_row(
        "FirstDifferenceAbove4kHz_dB_per_bin",
        frame["v32_first_difference_mae_db_per_bin"].to_numpy(),
        frame["v34a_first_difference_mae_db_per_bin"].to_numpy(),
    )
    second_statistics = paired_row(
        "SecondDifferenceAbove4kHz_dB_per_bin2",
        frame["v32_second_difference_mae_db_per_bin2"].to_numpy(),
        frame["v34a_second_difference_mae_db_per_bin2"].to_numpy(),
    )
    return rows, {
        "definition": {
            "error": "predicted_residual_db - target_residual_db",
            "minimum_frequency_hz": SPECTRAL_CUTOFF_HZ,
            "direction_subset": "all interpolation directions",
            "spatial_weighting": "normalized solid-angle weights per subject",
            "subject_aggregation": "equal mean over 44 validation subjects",
        },
        "first_difference": first_statistics,
        "second_difference": second_statistics,
    }


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
        squared_delta += float(torch.sum(delta * delta).item())
        squared_source += float(
            torch.sum(source_value.double() * source_value.double()).item()
        )
        maximum_delta = max(maximum_delta, float(torch.max(torch.abs(delta)).item()))
        tensor_count += 1
        parameter_count += source_value.numel()
    if tensor_count == 0:
        raise ValueError(f"No floating tensors found for prefix {prefix}")
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "maximum_absolute_parameter_delta": maximum_delta,
        "l2_parameter_delta": float(np.sqrt(squared_delta)),
        "relative_l2_parameter_delta": float(
            np.sqrt(squared_delta) / max(np.sqrt(squared_source), 1e-12)
        ),
        "bitwise_unchanged": bool(maximum_delta == 0.0),
    }


def checkpoint_diagnostics() -> dict[str, object]:
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    candidate = torch.load(
        CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    source_state = source["model_state"]
    candidate_state = candidate["model_state"]
    if source_state.keys() != candidate_state.keys():
        raise RuntimeError("Source and candidate model states have different keys")
    return {
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "candidate_checkpoint_sha256": sha256(CANDIDATE_CHECKPOINT),
        "candidate_checkpoint_epoch": int(candidate["epoch"]),
        "mlp": parameter_group_diagnostics(source_state, candidate_state, "mlp."),
        "local_cnn": parameter_group_diagnostics(source_state, candidate_state, "cnn."),
    }


def main() -> None:
    summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    if summary["split"] != "val" or summary["test_subject_count_read"] != 0:
        raise RuntimeError("v3.4a analysis must remain validation-only")

    comparison_rows: list[dict[str, object]] = []
    for aggregate in summary["aggregate"]:
        comparison_rows.append(
            {
                "metric": aggregate["Metric"],
                "v32_epoch39_mean_db": aggregate["MLPCNNv3Mean_dB"],
                "v34a_epoch7_mean_db": aggregate["MLPCNNv31Mean_dB"],
                "improvement_vs_v32_percent": aggregate[
                    "MLPCNNv31ImprovementVsMLPCNNv3_percent"
                ],
                "v34a_better_subject_count": aggregate[
                    "MLPCNNv31ImprovedVsMLPCNNv3_SubjectCount"
                ],
                "subject_count": aggregate["SubjectCount"],
            }
        )
    write_csv(RESULT_ROOT / "comparison_vs_v32_epoch39.csv", comparison_rows)

    per_subject = pd.read_csv(RESULT_ROOT / "per_subject_metrics.csv")
    subject_labels = per_subject["SubjectLabel"].tolist()
    if len(subject_labels) != 44 or len(set(subject_labels)) != 44:
        raise RuntimeError("Expected exactly 44 unique validation subjects")
    statistics_rows = paired_statistics(per_subject)
    write_csv(RESULT_ROOT / "paired_statistics.csv", statistics_rows)

    spectral_rows, spectral_summary = spectral_difference_diagnostics(subject_labels)
    write_csv(RESULT_ROOT / "spectral_difference_per_subject.csv", spectral_rows)
    write_csv(
        RESULT_ROOT / "spectral_difference_paired_statistics.csv",
        [spectral_summary["first_difference"], spectral_summary["second_difference"]],
    )

    training_report = json.loads(
        (CANDIDATE_TRAINING_ROOT / "training_report.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = {
        "schema_version": "1.0",
        "split": "val",
        "test_subject_count_read": 0,
        "subject_count": len(subject_labels),
        "spectral_difference": spectral_summary,
        "checkpoint": checkpoint_diagnostics(),
        "training": training_report,
    }
    (RESULT_ROOT / "spectral_difference_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
