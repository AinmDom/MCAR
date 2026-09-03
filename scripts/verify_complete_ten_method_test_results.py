"""Independently verify the consolidated 10-method x 20-endpoint test package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "sonicom_complete_ten_method_test_v1"
PRIMARY = ROOT / "results" / "sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method" / "metric_long.csv"
SUPPLEMENTARY = ROOT / "results" / "sonicom_complete_ten_method_secondary_deferred_test_v1" / "per_subject_metrics.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(items: list[dict[str, str]], value_column: str, endpoint_column: str) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for row in items:
        key = (row["SubjectLabel"], row["Method"], row[endpoint_column])
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = float(row[value_column])
    return result


def main() -> None:
    subject = rows(OUTPUT / "per_subject_metrics.csv")
    aggregate = rows(OUTPUT / "aggregate_metrics.csv")
    paired = rows(OUTPUT / "paired_vs_bounded.csv")
    wide = rows(OUTPUT / "paper_complete_test_wide.csv")
    index = rows(OUTPUT / "figure_index.csv")
    registry = rows(OUTPUT / "method_registry.csv")
    band = rows(OUTPUT / "band_ild_profile.csv")
    spatial = rows(OUTPUT / "spatial_direction_map.csv")
    if (len(subject), len(aggregate), len(paired), len(wide), len(index), len(registry), len(band), len(spatial)) != (8800, 200, 180, 20, 20, 10, 350, 7670):
        raise ValueError("Unexpected result-table dimensions")
    if [row["Method"] for row in registry] != [
        "SHOnly", "SUpDEqSH", "SUpDEqNN", "SUpDEqBary", "MCA",
        "MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED",
    ] or set(row["Status"] for row in registry) != {"COMPLETE"}:
        raise ValueError("Complete method registry mismatch")
    combined = keyed(subject, "Value", "Endpoint")
    if len(combined) != 8800:
        raise ValueError("Consolidated subject keys are not unique")
    primary = keyed(rows(PRIMARY), "Value_dB", "Metric")
    supplementary = keyed(rows(SUPPLEMENTARY), "Value", "Endpoint")
    if set(combined) != set(primary) | set(supplementary):
        raise ValueError("Source/consolidated key coverage mismatch")
    primary_error = max(abs(combined[key] - value) for key, value in primary.items())
    supplementary_error = max(abs(combined[key] - value) for key, value in supplementary.items())
    if primary_error != 0.0 or supplementary_error != 0.0:
        raise ValueError("Consolidation changed source values")
    cell_counts: dict[tuple[str, str], int] = {}
    for row in subject:
        key = (row["Method"], row["Endpoint"])
        cell_counts[key] = cell_counts.get(key, 0) + 1
    if len(cell_counts) != 200 or set(cell_counts.values()) != {44}:
        raise ValueError("Every method-endpoint cell must contain 44 subjects")
    numeric_columns = {
        "aggregate": (aggregate, ["Mean", "SampleStd", "Bootstrap95Lower", "Bootstrap95Upper"]),
        "paired": (paired, ["MeanDifference", "Bootstrap95Lower", "Bootstrap95Upper", "WinRate"]),
        "subject": (subject, ["Value"]),
    }
    for name, (table, columns) in numeric_columns.items():
        array = np.asarray([[float(row[column]) for column in columns] for row in table])
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"Non-finite {name} table")
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in subject:
        grouped.setdefault((row["Method"], row["Endpoint"]), []).append(float(row["Value"]))
    max_mean_error = max(abs(float(row["Mean"]) - np.mean(grouped[(row["Method"], row["Endpoint"])])) for row in aggregate)
    if max_mean_error > 1e-12:
        raise ValueError("Aggregate mean mismatch")
    if any(int(row["Wins"]) + int(row["Ties"]) + int(row["Losses"]) != 44 for row in paired):
        raise ValueError("Paired count mismatch")
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or not summary["all_finite"]:
        raise ValueError("Summary is not complete")
    report = {
        "status": "verified", "split": "test", "method_count": 10,
        "endpoint_count": 20, "subject_count": 44, "per_subject_rows": 8800,
        "aggregate_rows": 200, "paired_rows": 180, "wide_rows": 20, "method_registry_rows": 10,
        "all_numeric_finite": True, "primary_source_max_abs_error": primary_error,
        "supplementary_source_max_abs_error": supplementary_error,
        "aggregate_mean_max_abs_error": float(max_mean_error),
        "new_test_subject_count_read": 0,
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
