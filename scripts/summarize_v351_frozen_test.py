"""Add nonparametric and multiplicity-aware statistics to the MATLAB summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from mcar.paths import project_root


ROOT = project_root()
RESULT_ROOT = (
    ROOT / "results" / "sonicom_mlp_cnn_q26_v351_vs_v32e39_frozen_test_strict"
)
METRICS = {
    "full_sphere_erb": ("MLPCNNv3FullSphereERB_dB", "MLPCNNv31FullSphereERB_dB"),
    "contralateral_25deg_erb": (
        "MLPCNNv3Contralateral25ERB_dB",
        "MLPCNNv31Contralateral25ERB_dB",
    ),
    "contralateral_high_frequency": (
        "MLPCNNv3ContralateralHighFrequency_dB",
        "MLPCNNv31ContralateralHighFrequency_dB",
    ),
    "horizontal_ild_mae": (
        "MLPCNNv3HorizontalILDMAE_dB",
        "MLPCNNv31HorizontalILDMAE_dB",
    ),
}


def holm_adjust(raw: list[float]) -> list[float]:
    count = len(raw)
    order = np.argsort(raw)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * raw[index]))
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    with (RESULT_ROOT / "per_subject_metrics.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        subjects = list(csv.DictReader(handle))
    if len(subjects) != 44 or len({row["SubjectID"] for row in subjects}) != 44:
        raise RuntimeError("Expected 44 unique test subjects")

    matlab_report = json.loads(
        (RESULT_ROOT / "paired_comparison.json").read_text(encoding="utf-8")
    )
    matlab_by_metric = {row["Metric"]: row for row in matlab_report["metrics"]}
    output: list[dict[str, object]] = []
    paired_t_raw: list[float] = []
    wilcoxon_raw: list[float] = []
    for name, (main_column, candidate_column) in METRICS.items():
        main_values = np.asarray([float(row[main_column]) for row in subjects])
        candidate_values = np.asarray(
            [float(row[candidate_column]) for row in subjects]
        )
        if not np.all(np.isfinite(main_values)) or not np.all(
            np.isfinite(candidate_values)
        ):
            raise RuntimeError(f"Nonfinite values in {name}")
        t_result = stats.ttest_rel(candidate_values, main_values)
        wilcoxon_result = stats.wilcoxon(candidate_values, main_values)
        matlab = matlab_by_metric[name]
        if not np.isclose(
            float(t_result.pvalue), float(matlab["PairedT_P"]), rtol=1e-9, atol=1e-15
        ):
            raise RuntimeError(f"MATLAB/SciPy paired-t mismatch for {name}")
        row = {
            "metric": name,
            "v32_epoch39_mean_db": float(np.mean(main_values)),
            "v351_mean_db": float(np.mean(candidate_values)),
            "mean_difference_db": float(np.mean(candidate_values - main_values)),
            "relative_improvement_percent": float(
                100.0
                * (np.mean(main_values) - np.mean(candidate_values))
                / np.mean(main_values)
            ),
            "improved_subject_count_of_44": int(
                np.sum(candidate_values < main_values)
            ),
            "paired_t_p": float(t_result.pvalue),
            "wilcoxon_p": float(wilcoxon_result.pvalue),
            "cohen_dz": float(matlab["CohenDz"]),
            "ci95_low_db": float(matlab["CI95Low_dB"]),
            "ci95_high_db": float(matlab["CI95High_dB"]),
        }
        output.append(row)
        paired_t_raw.append(row["paired_t_p"])
        wilcoxon_raw.append(row["wilcoxon_p"])

    for row, t_adjusted, wilcoxon_adjusted in zip(
        output, holm_adjust(paired_t_raw), holm_adjust(wilcoxon_raw)
    ):
        row["paired_t_holm_p"] = t_adjusted
        row["wilcoxon_holm_p"] = wilcoxon_adjusted

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "v351_frozen_vs_v32_epoch39_frozen",
        "split": "test",
        "test_subject_count_read": 44,
        "matlab_scipy_paired_t_crosscheck_passed": True,
        "metrics": output,
        "model_or_weight_updates_after_test": False,
    }
    (RESULT_ROOT / "paired_comparison_complete.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULT_ROOT / "paired_comparison_complete.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
