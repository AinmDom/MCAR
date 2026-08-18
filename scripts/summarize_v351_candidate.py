"""Compute paired validation statistics for v3.5.1 candidate versus v3.5."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from mcar.paths import project_root


ROOT = project_root()
RESULT_ROOT = ROOT / "results" / "sonicom_mlp_cnn_q26_v351_candidate_strict_validation"
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


def main() -> None:
    with (RESULT_ROOT / "per_subject_metrics.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 44:
        raise RuntimeError(f"Expected 44 validation subjects, found {len(rows)}")

    output = []
    for name, (baseline_column, candidate_column) in METRICS.items():
        baseline = np.asarray([float(row[baseline_column]) for row in rows])
        candidate = np.asarray([float(row[candidate_column]) for row in rows])
        difference = candidate - baseline
        mean_difference = float(np.mean(difference))
        standard_deviation = float(np.std(difference, ddof=1))
        standard_error = standard_deviation / np.sqrt(len(rows))
        critical = float(stats.t.ppf(0.975, len(rows) - 1))
        output.append(
            {
                "metric": name,
                "v35_mean_db": float(np.mean(baseline)),
                "v351_candidate_mean_db": float(np.mean(candidate)),
                "mean_difference_db": mean_difference,
                "relative_improvement_percent": float(
                    100.0 * (np.mean(baseline) - np.mean(candidate)) / np.mean(baseline)
                ),
                "improved_subject_count_of_44": int(np.sum(candidate < baseline)),
                "ci95_low_db": mean_difference - critical * standard_error,
                "ci95_high_db": mean_difference + critical * standard_error,
                "paired_t_p": float(stats.ttest_rel(candidate, baseline).pvalue),
                "wilcoxon_p": float(stats.wilcoxon(candidate, baseline).pvalue),
                "cohen_dz": mean_difference / standard_deviation,
            }
        )

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "v351_candidate_vs_v35",
        "validation_subject_count": 44,
        "metrics": output,
        "test_subject_count_read": 0,
    }
    (RESULT_ROOT / "paired_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULT_ROOT / "paired_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
