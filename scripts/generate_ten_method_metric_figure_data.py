"""Create deterministic plotting summaries for the 24 registered ten-method metrics."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from generate_complete_ten_method_comparison_v2 import (
    DEFERRED_ENDPOINTS,
    LABELS,
    METHODS,
    PRIMARY_ENDPOINTS,
    SECONDARY_ENDPOINTS,
    UNITS,
    canonical_subject_rows,
    supplementary_means,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "sonicom_complete_ten_method_metric_figure_data_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260901

TIER_SPECS = (
    ("Primary validation", "val", PRIMARY_ENDPOINTS, "primary_validation"),
    (
        "Primary frozen engineering test",
        "test",
        PRIMARY_ENDPOINTS,
        "frozen_engineering_test",
    ),
    ("Secondary validation", "val", SECONDARY_ENDPOINTS, "secondary_validation"),
    ("Deferred validation", "val", DEFERRED_ENDPOINTS, "deferred_validation"),
)

ENDPOINT_TITLES = {
    "FullSphereERB": "Full-sphere ERB error",
    "Contralateral25ERB": "Contralateral 25-degree ERB error",
    "ContralateralHighFrequency": "Contralateral high-frequency error",
    "HorizontalILDMAE": "Horizontal-plane ILD MAE",
    "FullSphereLSD": "Full-sphere log-spectral distance",
    "HFFirstDifferenceMAE": "High-frequency first-difference MAE",
    "HFSecondDifferenceMAE": "High-frequency second-difference MAE",
    "MultiScaleNotchDepthMAE": "Multi-scale notch-depth MAE",
    "ERBBandILDMean": "ERB-band horizontal ILD error",
    "SpatialLSD_0_10deg": "Spatial LSD: 0-10 degrees from Q26",
    "SpatialLSD_10_20deg": "Spatial LSD: 10-20 degrees from Q26",
    "SpatialLSD_20_30deg": "Spatial LSD: 20-30 degrees from Q26",
    "SpatialLSD_30_180deg": "Spatial LSD: 30-180 degrees from Q26",
    "DominantNotchPenalizedMAE_Hz": "Dominant-notch penalized MAE",
    "DominantNotchMatchedMAE_Hz": "Dominant-notch matched-only MAE",
    "DominantNotchMissRate": "Dominant-notch miss rate",
    "DominantNotchSpuriousRate": "Dominant-notch spurious rate",
    "ReferenceNotchFraction": "Reference notch fraction",
    "ITDWeightedMAE_us": "ITD solid-angle-weighted MAE",
    "ITDMaximumAbsoluteError_us": "ITD maximum absolute error",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Inconsistent columns: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def normalized_rows() -> list[dict[str, object]]:
    validation, test = canonical_subject_rows()
    _, secondary, deferred = supplementary_means()
    rows: list[dict[str, object]] = []
    for source in (validation, test):
        for row in source:
            rows.append(
                {
                    "EvidenceTier": row["EvidenceTier"],
                    "Split": row["Split"],
                    "SubjectLabel": row["SubjectLabel"],
                    "Method": row["Method"],
                    "Endpoint": row["Endpoint"],
                    "Value": float(row["Value"]),
                }
            )
    for tier, source in (
        ("Secondary validation", secondary),
        ("Deferred validation", deferred),
    ):
        for row in source:
            rows.append(
                {
                    "EvidenceTier": tier,
                    "Split": "val",
                    "SubjectLabel": row["SubjectLabel"],
                    "Method": row["Method"],
                    "Endpoint": row["Endpoint"],
                    "Value": float(row["Value"]),
                }
            )
    return rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    rows = normalized_rows()
    cells: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["EvidenceTier"]), str(row["Endpoint"]), str(row["Method"]))
        subject = str(row["SubjectLabel"])
        if subject in cells[key]:
            raise ValueError(f"Duplicate subject cell: {key}/{subject}")
        value = float(row["Value"])
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite plot input: {key}/{subject}")
        cells[key][subject] = value

    expected = {
        (tier, endpoint, method)
        for tier, _, endpoints, _ in TIER_SPECS
        for endpoint in endpoints
        for method in METHODS
    }
    if set(cells) != expected:
        raise ValueError(
            f"Metric coverage mismatch; missing={sorted(expected - set(cells))}; "
            f"extra={sorted(set(cells) - expected)}"
        )
    if any(len(subjects) != 44 for subjects in cells.values()):
        raise ValueError("Every method-metric cell must contain 44 subjects")

    summary_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    figure_number = 0
    for tier_index, (tier, split, endpoints, tier_slug) in enumerate(TIER_SPECS):
        for endpoint_index, endpoint in enumerate(endpoints):
            figure_number += 1
            descriptive = endpoint == "ReferenceNotchFraction"
            percentage = endpoint in {
                "DominantNotchMissRate",
                "DominantNotchSpuriousRate",
                "ReferenceNotchFraction",
            }
            output_unit = "%" if percentage else UNITS[endpoint]
            figure_stem = f"{figure_number:02d}_{tier_slug}_{slug(endpoint)}"
            index_rows.append(
                {
                    "FigureNumber": figure_number,
                    "EvidenceTier": tier,
                    "Split": split,
                    "Endpoint": endpoint,
                    "Title": ENDPOINT_TITLES[endpoint],
                    "DisplayUnit": output_unit,
                    "Direction": "Descriptive only" if descriptive else "Lower is better",
                    "PngFile": f"png/{figure_stem}.png",
                    "PdfFile": f"pdf/{figure_stem}.pdf",
                }
            )
            for method_index, method in enumerate(METHODS):
                values = np.asarray(
                    [cells[(tier, endpoint, method)][subject] for subject in sorted(cells[(tier, endpoint, method)])],
                    dtype=np.float64,
                )
                if percentage:
                    values = 100.0 * values
                seed = (
                    BOOTSTRAP_SEED
                    + tier_index * 100_000
                    + endpoint_index * 1_000
                    + method_index
                )
                rng = np.random.Generator(np.random.PCG64(seed))
                draws = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
                bootstrap = values[draws].mean(axis=1)
                summary_rows.append(
                    {
                        "FigureNumber": figure_number,
                        "FigureStem": figure_stem,
                        "EvidenceTier": tier,
                        "Split": split,
                        "Endpoint": endpoint,
                        "Title": ENDPOINT_TITLES[endpoint],
                        "DisplayUnit": output_unit,
                        "Direction": "Descriptive only" if descriptive else "Lower is better",
                        "MethodOrder": method_index + 1,
                        "Method": method,
                        "MethodLabel": LABELS[method],
                        "SubjectCount": 44,
                        "Mean": format(float(values.mean()), ".17g"),
                        "SampleStd": format(float(values.std(ddof=1)), ".17g"),
                        "Bootstrap95Lower": format(float(np.quantile(bootstrap, 0.025)), ".17g"),
                        "Bootstrap95Upper": format(float(np.quantile(bootstrap, 0.975)), ".17g"),
                        "BootstrapReplicates": BOOTSTRAP_REPLICATES,
                        "BootstrapSeed": seed,
                    }
                )

    if len(summary_rows) != 24 * 10 or len(index_rows) != 24:
        raise ValueError("Unexpected plotting-table dimensions")
    write_csv(OUTPUT / "metric_summary.csv", summary_rows)
    write_csv(OUTPUT / "figure_index.csv", index_rows)
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "figure_count": 24,
        "method_count_per_figure": 10,
        "subject_count_per_method": 44,
        "summary_rows": len(summary_rows),
        "all_finite": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_SEED,
        "new_test_subject_count_read": 0,
        "source": "committed per-subject CSV results only",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
