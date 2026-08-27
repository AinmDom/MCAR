"""Apply pre-registered four-method validation summaries.

Reads the strict four-method metric_long.csv (44 subjects x 4 methods x 4
metrics), computes aggregate means and paired percentile 95% CIs between the
new FiLM-SIREN D1/D2+notch E130 ensemble and each baseline (MCAR v3.5.1,
RANF, FSP-AE) with 10000 replicates, and writes a decision JSON. Validation
only; test subjects are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20260819
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_SUBJECTS = 44

METHODS = ("FILM", "MCAR", "RANF", "FSPAE")
BASELINES = ("MCAR", "RANF", "FSPAE")
METRICS = (
    "FullSphereERB",
    "Contralateral25ERB",
    "ContralateralHighFrequency",
    "HorizontalILDMAE",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metric_long_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def paired_percentile_interval(
    differences: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
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
    with arguments.metric_long_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    subject_labels = sorted({row["SubjectLabel"] for row in rows})
    if len(subject_labels) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} unique subjects, found {len(subject_labels)}"
        )

    values: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["Method"], row["Metric"])
        if key not in values:
            values[key] = {}
        values[key][row["SubjectLabel"]] = float(row["Value_dB"])
    for method in METHODS:
        for metric in METRICS:
            if (method, metric) not in values:
                raise ValueError(f"Missing rows for {method}/{metric}")
            if len(values[(method, metric)]) != EXPECTED_SUBJECTS:
                raise ValueError(f"Incomplete rows for {method}/{metric}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aggregate: dict[str, dict[str, object]] = {}
    pairwise: dict[str, dict[str, object]] = {}
    for metric in METRICS:
        aggregate[metric] = {
            method: float(np.mean([values[(method, metric)][label] for label in subject_labels]))
            for method in METHODS
        }
    for metric in METRICS:
        for baseline in BASELINES:
            candidate_values = np.asarray(
                [values[("FILM", metric)][label] for label in subject_labels],
                dtype=np.float64,
            )
            baseline_values = np.asarray(
                [values[(baseline, metric)][label] for label in subject_labels],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(candidate_values)) or not np.all(
                np.isfinite(baseline_values)
            ):
                raise FloatingPointError(f"Non-finite values in {metric}/{baseline}")
            differences = candidate_values - baseline_values
            lower, upper = paired_percentile_interval(differences, rng)
            pairwise[f"{metric}__FILM_minus_{baseline}"] = {
                "definition": (
                    "FILM D1/D2+notch E130 ensemble minus "
                    f"{baseline}; negative is better"
                ),
                "film_mean_db": float(candidate_values.mean()),
                f"{baseline.lower()}_mean_db": float(baseline_values.mean()),
                "paired_mean_difference_db": float(differences.mean()),
                "paired_percentile_95_ci_db": [lower, upper],
                "film_wins": int(np.sum(candidate_values < baseline_values)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            }

    output = {
        "schema_version": "1.0",
        "analysis_scope": (
            "Validation-only four-method comparison; no test subjects read"
        ),
        "subject_count": EXPECTED_SUBJECTS,
        "methods": list(METHODS),
        "metrics": list(METRICS),
        "bootstrap": {
            "paired_resampling_unit": "subject row",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95% CI",
        },
        "aggregate": aggregate,
        "pairwise_film_vs_baseline": pairwise,
        "test_subject_count_read": 0,
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
