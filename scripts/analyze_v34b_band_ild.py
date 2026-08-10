"""Summarize the validation-only v3.4b spectral-band ILD ablation."""

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

from mcar.losses import spectral_band_ild_smooth_l1_and_mae
from mcar.training.train_mlp_v2 import (
    make_erb_center_frequencies_hz,
    make_erb_weights,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "sonicom_mlp_cnn_q26_v34b_band_ild_e10_strict_validation"
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
    / "sonicom_q26_validation_mlp_cnn_v34b_band_ild_e10"
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
    / "sonicom_mlp_cnn_q26_v34b_band_ild_e10"
)
CANDIDATE_CHECKPOINT = CANDIDATE_TRAINING_ROOT / "best.pt"
MINIMUM_BAND_CENTER_HZ = 200.0
MAXIMUM_BAND_CENTER_HZ = 18_000.0
BETA_DB = 0.5


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
    return [
        paired_row(
            metric,
            per_subject[baseline_column].to_numpy(dtype=np.float64),
            per_subject[candidate_column].to_numpy(dtype=np.float64),
        )
        for metric, (baseline_column, candidate_column) in METRICS.items()
    ]


def band_ild_metrics(
    corrected_db: np.ndarray,
    reference_db: np.ndarray,
    direction_weights: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    erb_weights = make_erb_weights(frequency_hz)
    centers = make_erb_center_frequencies_hz(erb_weights.shape[0])
    band_mask = (
        (centers >= MINIMUM_BAND_CENTER_HZ)
        & (centers <= MAXIMUM_BAND_CENTER_HZ)
    )
    log_weights = torch.log(
        torch.clamp(torch.from_numpy(erb_weights), min=1e-12)
    ).view(1, 1, erb_weights.shape[0], erb_weights.shape[1])
    smooth_l1, mae = spectral_band_ild_smooth_l1_and_mae(
        torch.from_numpy(corrected_db),
        torch.from_numpy(reference_db),
        torch.from_numpy(direction_weights),
        log_weights,
        torch.from_numpy(centers),
        MINIMUM_BAND_CENTER_HZ,
        MAXIMUM_BAND_CENTER_HZ,
        BETA_DB,
    )

    power_scale = np.log(10.0) / 10.0
    corrected_power = np.exp(power_scale * corrected_db.astype(np.float64))
    reference_power = np.exp(power_scale * reference_db.astype(np.float64))
    corrected_band_db = 10.0 * np.log10(
        np.einsum("edf,bf->edb", corrected_power, erb_weights)
    )
    reference_band_db = 10.0 * np.log10(
        np.einsum("edf,bf->edb", reference_power, erb_weights)
    )
    corrected_ild = corrected_band_db[0] - corrected_band_db[1]
    reference_ild = reference_band_db[0] - reference_band_db[1]
    normalized_weights = direction_weights / np.sum(direction_weights)
    per_band_mae = np.sum(
        np.abs(corrected_ild - reference_ild) * normalized_weights[:, None],
        axis=0,
    )[band_mask]
    return (
        float(smooth_l1),
        float(mae),
        per_band_mae,
        centers[band_mask],
    )


def spectral_band_diagnostics(
    subject_labels: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_by_band: list[np.ndarray] = []
    candidate_by_band: list[np.ndarray] = []
    retained_centers: np.ndarray | None = None
    for subject_label in subject_labels:
        source_path = DATASET_ROOT / "subjects" / subject_label / "q26.h5"
        baseline_path = BASELINE_ROOT / "subjects" / subject_label / "prediction.h5"
        candidate_path = CANDIDATE_ROOT / "subjects" / subject_label / "prediction.h5"
        with h5py.File(source_path, "r") as source_handle:
            direction_features = np.asarray(
                source_handle["direction_features"], dtype=np.float32
            )
            interpolation_mask = np.squeeze(
                np.asarray(
                    source_handle["interpolation_evaluation_mask"], dtype=np.uint8
                )
            ).astype(bool)
            horizontal_mask = interpolation_mask & (
                np.abs(direction_features[:, 1]) <= 1e-6
            )
            mca = np.asarray(
                source_handle["mca_logmag_db"][:, horizontal_mask, :],
                dtype=np.float32,
            )
            target = np.asarray(
                source_handle["target_residual_db"][:, horizontal_mask, :],
                dtype=np.float32,
            )
            frequency_hz = np.squeeze(
                np.asarray(source_handle["frequency_hz"], dtype=np.float32)
            )
            direction_weights = direction_features[horizontal_mask, 5]
        with h5py.File(baseline_path, "r") as baseline_handle:
            baseline = np.asarray(
                baseline_handle["predicted_residual_db"][:, horizontal_mask, :],
                dtype=np.float32,
            )
        with h5py.File(candidate_path, "r") as candidate_handle:
            candidate = np.asarray(
                candidate_handle["predicted_residual_db"][:, horizontal_mask, :],
                dtype=np.float32,
            )

        reference_db = mca + target
        baseline_smooth, baseline_mae, baseline_band, centers = band_ild_metrics(
            mca + baseline,
            reference_db,
            direction_weights,
            frequency_hz,
        )
        candidate_smooth, candidate_mae, candidate_band, _ = band_ild_metrics(
            mca + candidate,
            reference_db,
            direction_weights,
            frequency_hz,
        )
        if retained_centers is None:
            retained_centers = centers
        elif not np.array_equal(retained_centers, centers):
            raise RuntimeError("ERB-band centers changed across subjects")
        baseline_by_band.append(baseline_band)
        candidate_by_band.append(candidate_band)
        rows.append(
            {
                "subject_label": subject_label,
                "horizontal_interpolation_direction_count": int(
                    np.sum(horizontal_mask)
                ),
                "v32_spectral_band_ild_smooth_l1_db": baseline_smooth,
                "v34b_spectral_band_ild_smooth_l1_db": candidate_smooth,
                "smooth_l1_improvement_percent": 100.0
                * (baseline_smooth - candidate_smooth)
                / baseline_smooth,
                "v32_spectral_band_ild_mae_db": baseline_mae,
                "v34b_spectral_band_ild_mae_db": candidate_mae,
                "mae_improvement_percent": 100.0
                * (baseline_mae - candidate_mae)
                / baseline_mae,
            }
        )

    frame = pd.DataFrame(rows)
    smooth_statistics = paired_row(
        "HorizontalSpectralBandILDSmoothL1_dB",
        frame["v32_spectral_band_ild_smooth_l1_db"].to_numpy(),
        frame["v34b_spectral_band_ild_smooth_l1_db"].to_numpy(),
    )
    mae_statistics = paired_row(
        "HorizontalSpectralBandILDMAE_dB",
        frame["v32_spectral_band_ild_mae_db"].to_numpy(),
        frame["v34b_spectral_band_ild_mae_db"].to_numpy(),
    )
    baseline_band_array = np.stack(baseline_by_band)
    candidate_band_array = np.stack(candidate_by_band)
    assert retained_centers is not None
    frequency_rows = []
    for index, center in enumerate(retained_centers):
        baseline_values = baseline_band_array[:, index]
        candidate_values = candidate_band_array[:, index]
        frequency_rows.append(
            {
                "band_center_hz": float(center),
                "v32_mean_ild_mae_db": float(np.mean(baseline_values)),
                "v34b_mean_ild_mae_db": float(np.mean(candidate_values)),
                "improvement_percent": float(
                    100.0
                    * (np.mean(baseline_values) - np.mean(candidate_values))
                    / np.mean(baseline_values)
                ),
                "v34b_better_subject_count": int(
                    np.sum(candidate_values < baseline_values)
                ),
                "subject_count": len(subject_labels),
            }
        )
    summary = {
        "definition": {
            "band_count": int(len(retained_centers)),
            "minimum_band_center_hz": MINIMUM_BAND_CENTER_HZ,
            "maximum_band_center_hz": MAXIMUM_BAND_CENTER_HZ,
            "smooth_l1_beta_db": BETA_DB,
            "direction_subset": "all horizontal interpolation directions",
            "spatial_weighting": "normalized solid-angle weights per subject",
            "subject_aggregation": "equal mean over 44 validation subjects",
        },
        "smooth_l1": smooth_statistics,
        "mae": mae_statistics,
        "improved_band_count": int(
            np.sum(np.mean(candidate_band_array, axis=0) < np.mean(baseline_band_array, axis=0))
        ),
    }
    return rows, frequency_rows, summary


def parameter_group_diagnostics(
    source_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, object]:
    squared_delta = 0.0
    squared_source = 0.0
    maximum_delta = 0.0
    parameter_count = 0
    tensor_count = 0
    for key, source_value in source_state.items():
        if not key.startswith(prefix) or not torch.is_floating_point(source_value):
            continue
        delta = candidate_state[key].double() - source_value.double()
        squared_delta += float(torch.sum(delta * delta))
        squared_source += float(torch.sum(source_value.double().square()))
        maximum_delta = max(maximum_delta, float(torch.max(torch.abs(delta))))
        parameter_count += source_value.numel()
        tensor_count += 1
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
        raise RuntimeError("v3.4b analysis must remain validation-only")
    comparison_rows = [
        {
            "metric": aggregate["Metric"],
            "v32_epoch39_mean_db": aggregate["MLPCNNv3Mean_dB"],
            "v34b_epoch7_mean_db": aggregate["MLPCNNv31Mean_dB"],
            "improvement_vs_v32_percent": aggregate[
                "MLPCNNv31ImprovementVsMLPCNNv3_percent"
            ],
            "v34b_better_subject_count": aggregate[
                "MLPCNNv31ImprovedVsMLPCNNv3_SubjectCount"
            ],
            "subject_count": aggregate["SubjectCount"],
        }
        for aggregate in summary["aggregate"]
    ]
    write_csv(RESULT_ROOT / "comparison_vs_v32_epoch39.csv", comparison_rows)

    per_subject = pd.read_csv(RESULT_ROOT / "per_subject_metrics.csv")
    subject_labels = per_subject["SubjectLabel"].tolist()
    if len(subject_labels) != 44 or len(set(subject_labels)) != 44:
        raise RuntimeError("Expected exactly 44 unique validation subjects")
    strict_statistics = paired_statistics(per_subject)
    write_csv(RESULT_ROOT / "paired_statistics.csv", strict_statistics)

    band_rows, frequency_rows, band_summary = spectral_band_diagnostics(
        subject_labels
    )
    write_csv(RESULT_ROOT / "spectral_band_ild_per_subject.csv", band_rows)
    write_csv(
        RESULT_ROOT / "spectral_band_ild_paired_statistics.csv",
        [band_summary["smooth_l1"], band_summary["mae"]],
    )
    write_csv(
        RESULT_ROOT / "spectral_band_ild_by_center_frequency.csv",
        frequency_rows,
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
        "spectral_band_ild": band_summary,
        "checkpoint": checkpoint_diagnostics(),
        "training": training_report,
    }
    (RESULT_ROOT / "spectral_band_ild_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
