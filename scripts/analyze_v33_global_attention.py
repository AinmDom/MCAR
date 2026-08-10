"""Summarize the validation-only v3.3 global-context ablation."""

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
    / "sonicom_mlp_cnn_q26_v33_global_attention_e10_strict_validation"
)
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
    / "sonicom_q26_validation_mlp_cnn_v33_global_attention_e10"
)
SOURCE_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_seed20260809_e40"
    / "best.pt"
)
CANDIDATE_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v33_global_attention_e10"
    / "best.pt"
)


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(per_subject: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, (baseline_column, candidate_column) in METRICS.items():
        baseline = per_subject[baseline_column].to_numpy(dtype=np.float64)
        candidate = per_subject[candidate_column].to_numpy(dtype=np.float64)
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
        rows.append(
            {
                "metric": metric,
                "subject_count": len(difference),
                "mean_candidate_minus_baseline_db": float(np.mean(difference)),
                "ci95_low_db": float(interval[0]),
                "ci95_high_db": float(interval[1]),
                "paired_t_p_value": float(t_result.pvalue),
                "wilcoxon_p_value": float(wilcoxon.pvalue),
                "cohen_dz": float(
                    np.mean(difference) / np.std(difference, ddof=1)
                ),
                "candidate_better_subject_count": int(np.sum(difference < 0.0)),
            }
        )
    return rows


def prediction_delta_diagnostics(subject_labels: list[str]) -> dict[str, object]:
    all_differences: list[np.ndarray] = []
    frequency_absolute_sum: np.ndarray | None = None
    frequency_signed_sum: np.ndarray | None = None
    frequency_sample_count = 0
    frequency_hz: np.ndarray | None = None
    for subject_label in subject_labels:
        baseline_path = (
            BASELINE_ROOT / "subjects" / subject_label / "prediction.h5"
        )
        candidate_path = (
            CANDIDATE_ROOT / "subjects" / subject_label / "prediction.h5"
        )
        with h5py.File(baseline_path, "r") as baseline_handle:
            baseline = np.asarray(
                baseline_handle["predicted_residual_db"], dtype=np.float32
            )
        with h5py.File(candidate_path, "r") as candidate_handle:
            candidate = np.asarray(
                candidate_handle["predicted_residual_db"], dtype=np.float32
            )
        difference = candidate - baseline
        all_differences.append(difference.reshape(-1))
        absolute_by_frequency = np.sum(
            np.abs(difference), axis=(0, 1), dtype=np.float64
        )
        signed_by_frequency = np.sum(difference, axis=(0, 1), dtype=np.float64)
        if frequency_absolute_sum is None:
            frequency_absolute_sum = absolute_by_frequency
            frequency_signed_sum = signed_by_frequency
        else:
            frequency_absolute_sum += absolute_by_frequency
            assert frequency_signed_sum is not None
            frequency_signed_sum += signed_by_frequency
        frequency_sample_count += difference.shape[0] * difference.shape[1]
        if frequency_hz is None:
            source_path = (
                ROOT
                / "data"
                / "processed"
                / "sonicom_residual_q26_v1"
                / "subjects"
                / subject_label
                / "q26.h5"
            )
            with h5py.File(source_path, "r") as source_handle:
                frequency_hz = np.squeeze(
                    np.asarray(source_handle["frequency_hz"], dtype=np.float64)
                )

    difference = np.concatenate(all_differences)
    assert frequency_hz is not None
    assert frequency_absolute_sum is not None
    assert frequency_signed_sum is not None
    high_frequency_mask = (frequency_hz >= 10_000.0) & (frequency_hz <= 20_000.0)
    frequency_rows = [
        {
            "frequency_hz": float(frequency),
            "mean_absolute_global_delta_db": float(absolute_sum / frequency_sample_count),
            "mean_signed_global_delta_db": float(signed_sum / frequency_sample_count),
        }
        for frequency, absolute_sum, signed_sum in zip(
            frequency_hz,
            frequency_absolute_sum,
            frequency_signed_sum,
        )
    ]
    write_csv(RESULT_ROOT / "global_delta_by_frequency.csv", frequency_rows)
    return {
        "sample_count": int(difference.size),
        "mean_absolute_global_delta_db": float(np.mean(np.abs(difference))),
        "root_mean_square_global_delta_db": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "mean_signed_global_delta_db": float(np.mean(difference)),
        "median_absolute_global_delta_db": float(np.median(np.abs(difference))),
        "p95_absolute_global_delta_db": float(
            np.quantile(np.abs(difference), 0.95)
        ),
        "maximum_absolute_global_delta_db": float(np.max(np.abs(difference))),
        "mean_absolute_delta_10_20khz_db": float(
            np.mean(frequency_absolute_sum[high_frequency_mask])
            / frequency_sample_count
        ),
    }


def checkpoint_diagnostics() -> dict[str, object]:
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    candidate = torch.load(
        CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    source_state = source["model_state"]
    candidate_state = candidate["model_state"]
    frozen_differences = [
        torch.max(torch.abs(candidate_state[key] - value)).item()
        for key, value in source_state.items()
        if torch.is_floating_point(value)
    ]
    gate_bias = candidate_state["global_gate.bias"].float()
    gate_weight = candidate_state["global_gate.weight"].float()
    return {
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "candidate_checkpoint_sha256": sha256(CANDIDATE_CHECKPOINT),
        "frozen_mlp_local_cnn_max_absolute_parameter_delta": float(
            max(frozen_differences)
        ),
        "gate_bias": gate_bias.tolist(),
        "gate_bias_sigmoid": torch.sigmoid(gate_bias).tolist(),
        "gate_weight_l2": float(torch.linalg.vector_norm(gate_weight).item()),
        "gate_weight_max_absolute": float(torch.max(torch.abs(gate_weight)).item()),
    }


def main() -> None:
    summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    if summary["split"] != "val" or summary["test_subject_count_read"] != 0:
        raise RuntimeError("v3.3 analysis must remain validation-only")
    comparison_rows: list[dict[str, object]] = []
    for aggregate in summary["aggregate"]:
        comparison_rows.append(
            {
                "metric": aggregate["Metric"],
                "v32_epoch39_mean_db": aggregate["MLPCNNv3Mean_dB"],
                "v33_epoch9_mean_db": aggregate["MLPCNNv31Mean_dB"],
                "improvement_vs_v32_percent": aggregate[
                    "MLPCNNv31ImprovementVsMLPCNNv3_percent"
                ],
                "v33_better_subject_count": aggregate[
                    "MLPCNNv31ImprovedVsMLPCNNv3_SubjectCount"
                ],
                "subject_count": aggregate["SubjectCount"],
            }
        )
    write_csv(RESULT_ROOT / "comparison_vs_v32_epoch39.csv", comparison_rows)

    per_subject = pd.read_csv(RESULT_ROOT / "per_subject_metrics.csv")
    statistics_rows = paired_statistics(per_subject)
    write_csv(RESULT_ROOT / "paired_statistics.csv", statistics_rows)

    diagnostics = {
        "schema_version": "1.0",
        "split": "val",
        "test_subject_count_read": 0,
        "prediction_delta": prediction_delta_diagnostics(
            per_subject["SubjectLabel"].tolist()
        ),
        "checkpoint": checkpoint_diagnostics(),
    }
    (RESULT_ROOT / "global_context_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
