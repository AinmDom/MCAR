"""Merge scratch and output-fusion validation results into the prior study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from mcar.paths import project_root


ROOT = project_root()
STUDY_ROOT = (
    ROOT / "results" / "sonicom_mlp_cnn_q26_v32_cnn_reinit_joint_v1"
)
PREVIOUS_ROOT = (
    ROOT
    / "results"
    / "sonicom_mlp_cnn_q26_v321_cnn_reinit_joint_unfreeze_e10_strict_validation"
)
SCRATCH_ROOT = (
    ROOT
    / "results"
    / "sonicom_mlp_cnn_q26_v32_scratch_joint_e60_strict_validation"
)
FUSION_ROOT = (
    ROOT
    / "results"
    / "sonicom_mlp_cnn_q26_v32_previous_scratch_fusion_strict_validation"
)
FUSION_REPORT = (
    ROOT
    / "results"
    / "sonicom_mlp_cnn_q26_v32_previous_scratch_fusion"
    / "fusion_report.json"
)
SCRATCH_TRAINING_REPORT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_scratch_joint_seed20260809_e60"
    / "training_report.json"
)


METRICS = {
    "full_sphere_erb": "FullSphereERB",
    "contralateral_25_erb": "Contralateral25ERB",
    "contralateral_hf": "ContralateralHighFrequency",
    "horizontal_ild": "HorizontalILDMAE",
}
SUBJECT_COLUMNS = {
    "full_sphere_erb": "MLPCNNv31FullSphereERB_dB",
    "contralateral_25_erb": "MLPCNNv31Contralateral25ERB_dB",
    "contralateral_hf": "MLPCNNv31ContralateralHighFrequency_dB",
    "horizontal_ild": "MLPCNNv31HorizontalILDMAE_dB",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def strict_means(root: Path) -> dict[str, float]:
    rows = read_csv(root / "aggregate_metrics.csv")
    by_metric = {row["Metric"]: row for row in rows}
    return {
        name: float(by_metric[metric]["MLPCNNv31Mean_dB"])
        for name, metric in METRICS.items()
    }


def subject_values(root: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(root / "per_subject_metrics.csv")
    return {
        row["SubjectLabel"]: {
            metric: float(row[column])
            for metric, column in SUBJECT_COLUMNS.items()
        }
        for row in rows
    }


def paired_rows(
    label: str,
    reference: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    subjects = sorted(reference)
    if subjects != sorted(candidate) or len(subjects) != 44:
        raise RuntimeError(f"Subject mismatch for {label}")
    output: list[dict[str, object]] = []
    for metric in METRICS:
        reference_values = np.asarray(
            [reference[subject][metric] for subject in subjects],
            dtype=np.float64,
        )
        candidate_values = np.asarray(
            [candidate[subject][metric] for subject in subjects],
            dtype=np.float64,
        )
        difference = candidate_values - reference_values
        mean_difference = float(np.mean(difference))
        standard_deviation = float(np.std(difference, ddof=1))
        standard_error = standard_deviation / np.sqrt(len(subjects))
        t_critical = float(stats.t.ppf(0.975, len(subjects) - 1))
        t_result = stats.ttest_rel(candidate_values, reference_values)
        wilcoxon = stats.wilcoxon(candidate_values, reference_values)
        output.append(
            {
                "comparison": label,
                "metric": metric,
                "mean_difference_db": mean_difference,
                "relative_improvement_percent": (
                    100.0
                    * (float(np.mean(reference_values)) - float(np.mean(candidate_values)))
                    / float(np.mean(reference_values))
                ),
                "improved_subject_count": int(
                    np.sum(candidate_values < reference_values)
                ),
                "ci95_low_db": mean_difference - t_critical * standard_error,
                "ci95_high_db": mean_difference + t_critical * standard_error,
                "paired_t_p": float(t_result.pvalue),
                "wilcoxon_p": float(wilcoxon.pvalue),
                "cohen_dz": mean_difference / standard_deviation,
            }
        )
    return output


def main() -> None:
    required = (
        STUDY_ROOT / "comparison.csv",
        STUDY_ROOT / "paired_statistics.csv",
        PREVIOUS_ROOT / "per_subject_metrics.csv",
        SCRATCH_ROOT / "per_subject_metrics.csv",
        FUSION_ROOT / "per_subject_metrics.csv",
        FUSION_REPORT,
        SCRATCH_TRAINING_REPORT,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    scratch_report = json.loads(SCRATCH_TRAINING_REPORT.read_text(encoding="utf-8"))
    fusion_report = json.loads(FUSION_REPORT.read_text(encoding="utf-8"))
    scratch = strict_means(SCRATCH_ROOT)
    fusion = strict_means(FUSION_ROOT)

    comparison = read_csv(STUDY_ROOT / "comparison.csv")
    new_names = {
        "v32_scratch_joint_epoch60",
        "fusion_previous30_scratch70",
        "v35_output_fusion_previous30_scratch70",
    }
    comparison = [row for row in comparison if row["method"] not in new_names]
    comparison.extend(
        [
            {
                "method": "v32_scratch_joint_epoch60",
                "validation_total_loss": scratch_report[
                    "best_validation_total_loss"
                ],
                "full_sphere_erb_db": scratch["full_sphere_erb"],
                "contralateral_25_erb_db": scratch["contralateral_25_erb"],
                "contralateral_high_frequency_db": scratch[
                    "contralateral_hf"
                ],
                "horizontal_ild_mae_db": scratch["horizontal_ild"],
            },
            {
                "method": "v35_output_fusion_previous30_scratch70",
                "validation_total_loss": fusion_report["best"]["total_loss"],
                "full_sphere_erb_db": fusion["full_sphere_erb"],
                "contralateral_25_erb_db": fusion["contralateral_25_erb"],
                "contralateral_high_frequency_db": fusion[
                    "contralateral_hf"
                ],
                "horizontal_ild_mae_db": fusion["horizontal_ild"],
            },
        ]
    )
    write_csv(STUDY_ROOT / "comparison.csv", comparison)

    previous_subjects = subject_values(PREVIOUS_ROOT)
    scratch_subjects = subject_values(SCRATCH_ROOT)
    fusion_subjects = subject_values(FUSION_ROOT)
    statistics = read_csv(STUDY_ROOT / "paired_statistics.csv")
    new_comparisons = {
        "scratch_vs_previous_joint",
        "fusion_vs_scratch",
        "fusion_vs_previous_joint",
    }
    statistics = [
        row for row in statistics if row["comparison"] not in new_comparisons
    ]
    statistics.extend(
        paired_rows(
            "scratch_vs_previous_joint", previous_subjects, scratch_subjects
        )
    )
    statistics.extend(
        paired_rows("fusion_vs_scratch", scratch_subjects, fusion_subjects)
    )
    statistics.extend(
        paired_rows(
            "fusion_vs_previous_joint", previous_subjects, fusion_subjects
        )
    )
    write_csv(STUDY_ROOT / "paired_statistics.csv", statistics)
    write_csv(
        STUDY_ROOT / "scratch_fusion_paired_statistics.csv",
        [row for row in statistics if row["comparison"] in new_comparisons],
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "scratch": scratch,
                "fusion": fusion,
                "comparison_rows": len(comparison),
                "paired_rows": len(statistics),
                "test_subject_count_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
