"""Apply the pre-registered FiLM-SIREN versus MCAR v3.5.1 test gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20260818
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_SUBJECTS = 44

METRICS = {
    "full_sphere_erb": {
        "baseline": "MLPCNNv3FullSphereERB_dB",
        "candidate": "MLPCNNv31FullSphereERB_dB",
        "margin_db": 0.0,
        "gate": "superiority",
    },
    "contralateral_25_erb": {
        "baseline": "MLPCNNv3Contralateral25ERB_dB",
        "candidate": "MLPCNNv31Contralateral25ERB_dB",
        "margin_db": 0.02,
        "gate": "non_inferiority",
    },
    "contralateral_high_frequency": {
        "baseline": "MLPCNNv3ContralateralHighFrequency_dB",
        "candidate": "MLPCNNv31ContralateralHighFrequency_dB",
        "margin_db": 0.05,
        "gate": "non_inferiority",
    },
    "strict_horizontal_ild": {
        "baseline": "MLPCNNv3HorizontalILDMAE_dB",
        "candidate": "MLPCNNv31HorizontalILDMAE_dB",
        "margin_db": 0.02,
        "gate": "non_inferiority",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("per_subject_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def paired_percentile_interval(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        differences.size,
        size=(BOOTSTRAP_REPLICATES, differences.size),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lower), float(upper)


def main() -> None:
    arguments = parse_arguments()
    with arguments.per_subject_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} paired rows, found {len(rows)}")
    subject_labels = [row["SubjectLabel"] for row in rows]
    if len(set(subject_labels)) != EXPECTED_SUBJECTS:
        raise ValueError("Subject labels are not unique")

    results: dict[str, dict[str, object]] = {}
    for name, specification in METRICS.items():
        baseline = np.asarray(
            [float(row[str(specification["baseline"])]) for row in rows],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [float(row[str(specification["candidate"])]) for row in rows],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
            raise FloatingPointError(f"Non-finite values in {name}")
        differences = candidate - baseline
        lower, upper = paired_percentile_interval(differences)
        margin = float(specification["margin_db"])
        results[name] = {
            "definition": "FiLM-SIREN minus MCAR v3.5.1; negative is better",
            "baseline_mean_db": float(baseline.mean()),
            "candidate_mean_db": float(candidate.mean()),
            "paired_mean_difference_db": float(differences.mean()),
            "paired_percentile_95_ci_db": [lower, upper],
            "candidate_wins": int(np.sum(candidate < baseline)),
            "margin_db": margin,
            "gate": specification["gate"],
            "passed": bool(upper < margin),
        }

    primary_wins = int(results["full_sphere_erb"]["candidate_wins"])
    primary_ci_passed = bool(results["full_sphere_erb"]["passed"])
    primary_distribution_passed = primary_wins >= 26
    secondary_passed = all(
        bool(results[name]["passed"])
        for name in (
            "contralateral_25_erb",
            "contralateral_high_frequency",
            "strict_horizontal_ild",
        )
    )
    overall_passed = primary_ci_passed and primary_distribution_passed and secondary_passed
    output = {
        "schema_version": "1.0",
        "analysis_scope": (
            "Frozen SONICOM engineering test; not an independent paper confirmation"
        ),
        "subject_count": EXPECTED_SUBJECTS,
        "bootstrap": {
            "paired_resampling_unit": "subject row",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95% CI",
        },
        "metrics": results,
        "gates": {
            "primary_superiority_upper_ci_below_zero": primary_ci_passed,
            "primary_full_erb_wins_at_least_26_of_44": primary_distribution_passed,
            "secondary_non_inferiority": secondary_passed,
            "overall_engineering_promotion": overall_passed,
        },
        "decision": "PROMOTE" if overall_passed else "DO_NOT_PROMOTE",
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
