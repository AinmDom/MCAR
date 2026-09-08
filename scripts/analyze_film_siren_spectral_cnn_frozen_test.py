"""Merge Hybrid E190 test metrics with the frozen nine-method test table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260831
REPLICATES = 10_000
METRICS = (
    "FullSphereERB",
    "Contralateral25ERB",
    "ContralateralHighFrequency",
    "HorizontalILDMAE",
)
BASELINES = ("BOUNDED", "MCARv351", "RANF", "FSPAE", "MCA")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_metric_long", type=Path)
    parser.add_argument("frozen_nine_metric_long", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate-label",
        default="FiLM-SIREN + spectral CNN Hybrid E190 1/3 ensemble",
        help="Display label for the newly evaluated MCARv32 candidate.",
    )
    arguments = parser.parse_args()

    raw = read(arguments.raw_metric_long)
    frozen = read(arguments.frozen_nine_metric_long)
    candidate = [
        dict(
            row,
            Method="HYBRID",
            MethodLabel=arguments.candidate_label,
        )
        for row in raw
        if row["Method"] == "MCARv32"
    ]
    if len(candidate) != 44 * 4:
        raise ValueError(f"Expected 176 candidate rows, found {len(candidate)}")

    shared = {row["Method"] for row in raw} & {row["Method"] for row in frozen}
    raw_map = {
        (row["SubjectLabel"], row["Method"], row["Metric"]): float(row["Value_dB"])
        for row in raw
        if row["Method"] in shared
    }
    frozen_map = {
        (row["SubjectLabel"], row["Method"], row["Metric"]): float(row["Value_dB"])
        for row in frozen
        if row["Method"] in shared
    }
    if raw_map.keys() != frozen_map.keys():
        raise ValueError("Shared frozen baseline keys differ")
    repeat_max = max((abs(raw_map[key] - frozen_map[key]) for key in raw_map), default=0.0)
    if repeat_max > 1e-10:
        raise ValueError(f"Frozen baseline reproduction mismatch: {repeat_max}")

    combined = candidate + frozen
    keys = {(row["SubjectLabel"], row["Method"], row["Metric"]) for row in combined}
    if len(combined) != 44 * 10 * 4 or len(keys) != len(combined):
        raise ValueError("Ten-method table is incomplete or duplicated")
    values = np.asarray([float(row["Value_dB"]) for row in combined])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite metric")

    arguments.output.mkdir(parents=True, exist_ok=False)
    fields = ["SubjectLabel", "SubjectID", "Method", "MethodLabel", "Metric", "Value_dB"]
    write(arguments.output / "metric_long.csv", combined, fields)
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in combined:
        groups[(row["Method"], row["MethodLabel"], row["Metric"])].append(
            float(row["Value_dB"])
        )
    aggregate = [
        {
            "Method": key[0],
            "MethodLabel": key[1],
            "Metric": key[2],
            "Mean_dB": float(np.mean(group_values)),
            "Std_dB": float(np.std(group_values, ddof=1)),
            "SubjectCount": len(group_values),
        }
        for key, group_values in sorted(groups.items())
    ]
    write(
        arguments.output / "aggregate.csv",
        aggregate,
        ["Method", "MethodLabel", "Metric", "Mean_dB", "Std_dB", "SubjectCount"],
    )

    lookup = {
        (row["SubjectLabel"], row["Method"], row["Metric"]): float(row["Value_dB"])
        for row in combined
    }
    subjects = sorted({row["SubjectLabel"] for row in candidate})
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 44, size=(REPLICATES, 44))
    paired = []
    for baseline in BASELINES:
        for metric in METRICS:
            difference = np.asarray(
                [
                    lookup[(subject, "HYBRID", metric)]
                    - lookup[(subject, baseline, metric)]
                    for subject in subjects
                ]
            )
            bootstrap = difference[indices].mean(axis=1)
            lower, upper = np.percentile(bootstrap, [2.5, 97.5])
            paired.append(
                {
                    "Candidate": "HYBRID",
                    "Baseline": baseline,
                    "Metric": metric,
                    "MeanDifference_dB": float(difference.mean()),
                    "CI95Lower_dB": float(lower),
                    "CI95Upper_dB": float(upper),
                    "CandidateWins": int(np.sum(difference < 0)),
                    "SubjectCount": 44,
                    "BootstrapReplicates": REPLICATES,
                    "BootstrapSeed": SEED,
                }
            )
    write(arguments.output / "paired_bootstrap.csv", paired, list(paired[0]))
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "candidate": "HYBRID",
        "subject_count": 44,
        "method_count": 10,
        "metric_count": 4,
        "metric_rows": len(combined),
        "all_finite": True,
        "raw_shared_baseline_max_abs_difference_db": repeat_max,
        "test_subject_count_read_by_candidate_inference": 44,
        "test_subject_count_read_by_merge": 0,
        "bootstrap": {"replicates": REPLICATES, "seed": SEED},
        "interpretation": "Results are final regardless of outcome; no test-driven tuning.",
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
