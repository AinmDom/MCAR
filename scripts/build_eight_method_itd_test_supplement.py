"""Derive the Q26 test ITD supplement for the requested eight-method subset.

This is a result-level extraction only. It reads the already completed frozen
ten-method supplementary CSV and does not access test subjects or predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "results" / "sonicom_complete_ten_method_secondary_deferred_test_v1"
OUTPUT = ROOT / "results" / "sonicom_eight_method_hybrid_itd_test_v1"
METHODS = (
    "SHOnly",
    "SUpDEqSH",
    "SUpDEqNN",
    "SUpDEqBary",
    "MCA",
    "FSPAE",
    "RANF",
    "HYBRID",
)
LABELS = {
    "SHOnly": "SH only",
    "SUpDEqSH": "SUpDEq SH",
    "SUpDEqNN": "SUpDEq NN",
    "SUpDEqBary": "SUpDEq Barycentric",
    "MCA": "MCA",
    "FSPAE": "FSP-AE",
    "RANF": "RANF",
    "HYBRID": "Hybrid E190",
}
ENDPOINTS = ("ITDWeightedMAE_us", "ITDMaximumAbsoluteError_us")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    source_rows = read_rows(SOURCE_ROOT / "per_subject_metrics.csv")
    selected = [
        row for row in source_rows
        if row["Method"] in METHODS and row["Endpoint"] in ENDPOINTS
    ]
    expected = 44 * len(METHODS) * len(ENDPOINTS)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} selected rows, got {len(selected)}")

    keys: set[tuple[str, str, str]] = set()
    for row in selected:
        key = (row["SubjectLabel"], row["Method"], row["Endpoint"])
        if key in keys:
            raise ValueError(f"Duplicate key: {key}")
        keys.add(key)
        if row["Unit"] != "us" or not np.isfinite(float(row["Value"])):
            raise FloatingPointError(f"Invalid ITD row: {row}")

    selected.sort(
        key=lambda row: (
            ENDPOINTS.index(row["Endpoint"]),
            METHODS.index(row["Method"]),
            int(row["SubjectID"]),
        )
    )
    aggregate_source = read_rows(SOURCE_ROOT / "aggregate_metrics.csv")
    aggregate = [
        {
            **row,
            "MethodOrder": str(METHODS.index(row["Method"]) + 1),
        }
        for row in aggregate_source
        if row["Method"] in METHODS and row["Endpoint"] in ENDPOINTS
    ]
    aggregate.sort(key=lambda row: (ENDPOINTS.index(row["Endpoint"]), int(row["MethodOrder"])))
    if len(aggregate) != len(METHODS) * len(ENDPOINTS):
        raise ValueError("Incomplete aggregate ITD coverage")
    if any(int(row["SubjectCount"]) != 44 for row in aggregate):
        raise ValueError("Aggregate ITD subject count is not 44")
    if any(not np.isfinite(float(row[column])) for row in aggregate for column in (
        "Mean", "SampleStd", "Bootstrap95Lower", "Bootstrap95Upper"
    )):
        raise FloatingPointError("Non-finite aggregate ITD value")

    wide: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        rows = [row for row in aggregate if row["Endpoint"] == endpoint]
        wide_row: dict[str, object] = {
            "Split": "test",
            "Condition": "SONICOM-Q26-v1",
            "Endpoint": endpoint,
            "Unit": "us",
            "Direction": "Lower is better",
        }
        for row in rows:
            wide_row[LABELS[row["Method"]]] = float(row["Mean"])
        wide.append(wide_row)

    OUTPUT.mkdir(parents=True)
    write_rows(OUTPUT / "metric_long.csv", selected)
    write_rows(OUTPUT / "aggregate_metrics.csv", aggregate)
    write_rows(OUTPUT / "paper_itd_wide.csv", wide)

    source_manifest = json.loads(
        (ROOT / "configs/experiments/sonicom_complete_ten_method_secondary_deferred_test_v1_manifest.json")
        .read_text(encoding="utf-8")
    )
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "test",
        "condition": "SONICOM-Q26-v1",
        "subject_count": 44,
        "method_count": len(METHODS),
        "methods": list(METHODS),
        "method_labels": [LABELS[method] for method in METHODS],
        "endpoint_count": len(ENDPOINTS),
        "endpoints": list(ENDPOINTS),
        "per_subject_rows": len(selected),
        "aggregate_rows": len(aggregate),
        "all_finite": True,
        "source_result_root": str(SOURCE_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_identity_sha256": source_manifest["identity_sha256"],
        "source_test_subject_count_read": 44,
        "new_test_subject_count_read_during_derivation": 0,
        "source_sha256": {
            "per_subject_metrics.csv": sha256(SOURCE_ROOT / "per_subject_metrics.csv"),
            "aggregate_metrics.csv": sha256(SOURCE_ROOT / "aggregate_metrics.csv"),
        },
        "method_selection_note": (
            "MCA is included as the eighth method to match the requested count; "
            "MCARv351 and BOUNDED are excluded."
        ),
        "claim_boundary": (
            "Result-level extraction from the completed post-lock supplementary test "
            "characterization; no new inference, tuning, or test-data read."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# SONICOM Q26 test eight-method ITD supplement\n\n"
        "This package extracts the two registered ITD endpoints from the completed "
        "ten-method post-lock test evaluation. The eight-method subset is SH only, "
        "SUpDEq SH/NN/Barycentric, MCA, FSP-AE, RANF, and Hybrid E190.\n\n"
        "Values are in microseconds (us), lower is better. No new test inference or "
        "raw test-subject read occurred during this extraction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
