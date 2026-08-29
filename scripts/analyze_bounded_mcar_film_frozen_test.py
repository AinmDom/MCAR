"""Merge the frozen candidate with the frozen eight-method test table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260829
REPLICATES = 10_000
METRICS = ("FullSphereERB", "Contralateral25ERB", "ContralateralHighFrequency", "HorizontalILDMAE")
BASELINES = ("MCARv351", "RANF", "FSPAE", "MCA")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_metric_long", type=Path)
    parser.add_argument("frozen_eight_metric_long", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw, frozen = read(args.raw_metric_long), read(args.frozen_eight_metric_long)
    candidate = [dict(row, Method="BOUNDED", MethodLabel="Bounded MCAR + FiLM correction E25 1/3 ensemble")
                 for row in raw if row["Method"] == "MCARv32"]
    if len(candidate) != 44 * 4:
        raise ValueError(f"Expected 176 candidate rows, found {len(candidate)}")
    shared = {row["Method"] for row in raw} & {row["Method"] for row in frozen}
    raw_map = {(r["SubjectLabel"], r["Method"], r["Metric"]): float(r["Value_dB"])
               for r in raw if r["Method"] in shared}
    frozen_map = {(r["SubjectLabel"], r["Method"], r["Metric"]): float(r["Value_dB"])
                  for r in frozen if r["Method"] in shared}
    if raw_map.keys() != frozen_map.keys():
        raise ValueError("Shared frozen baseline keys differ")
    repeat_max = max((abs(raw_map[k] - frozen_map[k]) for k in raw_map), default=0.0)
    if repeat_max > 1e-10:
        raise ValueError(f"Frozen baseline reproduction mismatch: {repeat_max}")
    combined = candidate + frozen
    keys = {(r["SubjectLabel"], r["Method"], r["Metric"]) for r in combined}
    if len(combined) != 44 * 9 * 4 or len(keys) != len(combined):
        raise ValueError("Nine-method table is incomplete or duplicated")
    values = np.asarray([float(r["Value_dB"]) for r in combined])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite metric")
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ["SubjectLabel", "SubjectID", "Method", "MethodLabel", "Metric", "Value_dB"]
    write(args.output / "metric_long.csv", combined, fields)
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in combined:
        groups[(row["Method"], row["MethodLabel"], row["Metric"])].append(float(row["Value_dB"]))
    aggregate = [{"Method": k[0], "MethodLabel": k[1], "Metric": k[2],
                  "Mean_dB": float(np.mean(v)), "Std_dB": float(np.std(v, ddof=1)), "SubjectCount": len(v)}
                 for k, v in sorted(groups.items())]
    write(args.output / "aggregate.csv", aggregate,
          ["Method", "MethodLabel", "Metric", "Mean_dB", "Std_dB", "SubjectCount"])
    lookup = {(r["SubjectLabel"], r["Method"], r["Metric"]): float(r["Value_dB"]) for r in combined}
    subjects = sorted({r["SubjectLabel"] for r in candidate})
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 44, size=(REPLICATES, 44))
    paired = []
    for baseline in BASELINES:
        for metric in METRICS:
            diff = np.asarray([lookup[(s, "BOUNDED", metric)] - lookup[(s, baseline, metric)] for s in subjects])
            boot = diff[indices].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            paired.append({"Candidate": "BOUNDED", "Baseline": baseline, "Metric": metric,
                           "MeanDifference_dB": float(diff.mean()), "CI95Lower_dB": float(lo),
                           "CI95Upper_dB": float(hi), "CandidateWins": int(np.sum(diff < 0)),
                           "SubjectCount": 44, "BootstrapReplicates": REPLICATES, "BootstrapSeed": SEED})
    write(args.output / "paired_bootstrap.csv", paired, list(paired[0]))
    summary = {"schema_version": "1.0", "status": "completed", "candidate": "BOUNDED",
               "subject_count": 44, "method_count": 9, "metric_count": 4,
               "metric_rows": len(combined), "all_finite": True,
               "raw_shared_baseline_max_abs_difference_db": repeat_max,
               "test_subject_count_read_by_candidate_inference": 44,
               "test_subject_count_read_by_merge": 0,
               "bootstrap": {"replicates": REPLICATES, "seed": SEED},
               "interpretation": "Results are final regardless of outcome; no test-driven tuning."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
