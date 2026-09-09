"""Compute unified subject-paired bootstrap statistics for five paper metrics.

This is a result-level derivation. It reads only frozen test CSV files and uses
the project's shared paired bootstrap implementation; it never loads models or
raw HRTF data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from mcar.evaluation.secondary_metrics import paired_tail_statistics


REPLICATES = 10_000
SEED = 20260906
CANDIDATE = "HYBRID"
BASELINES = ("MCA", "FSPAE", "RANF")
METHOD_LABELS = {
    CANDIDATE: "FSC/Hybrid E190",
    "MCA": "MCA",
    "FSPAE": "FSP-AE",
    "RANF": "RANF",
}
METRIC_SPECS = (
    {
        "metric": "FullSphereERB",
        "metric_label": "Full-sphere ERB",
        "unit": "dB",
        "evidence_tier": "Primary frozen engineering test",
        "source_kind": "metric_long",
        "value_column": "Value_dB",
    },
    {
        "metric": "Contralateral25ERB",
        "metric_label": "Contralateral-25 ERB",
        "unit": "dB",
        "evidence_tier": "Primary frozen engineering test",
        "source_kind": "metric_long",
        "value_column": "Value_dB",
    },
    {
        "metric": "ERBBandILDMean",
        "metric_label": "ERB-band ILD mean",
        "unit": "dB",
        "evidence_tier": "Secondary test",
        "source_kind": "per_subject",
        "value_column": "Value",
    },
    {
        "metric": "ITDWeightedMAE_us",
        "metric_label": "ITD weighted MAE",
        "unit": "us",
        "evidence_tier": "Deferred test",
        "source_kind": "per_subject",
        "value_column": "Value",
    },
    {
        "metric": "LAP2024LSD_dB",
        "metric_label": "LAP 2024 LSD",
        "unit": "dB",
        "evidence_tier": "LAP2024 locked-test descriptive evaluation",
        "source_kind": "lap",
        "value_column": "LAP2024LSD_dB",
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_unique_map(
    rows: list[dict[str, str]],
    *,
    key_fields: tuple[str, ...],
    value_field: str,
    label: str,
) -> dict[tuple[str, ...], float]:
    result: dict[tuple[str, ...], float] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError(f"Duplicate {label} key: {key}")
        try:
            value = float(row[value_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {label} value for {key}") from exc
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite {label} value for {key}")
        result[key] = value
    return result


def subject_set(values: Iterable[str]) -> set[str]:
    return set(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sonicom_fsc_hybrid_e190_five_metric_paired_bootstrap_v1"),
    )
    parser.add_argument(
        "--main-source",
        type=Path,
        default=Path(
            "results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/metric_long.csv"
        ),
    )
    parser.add_argument(
        "--supplementary-source",
        type=Path,
        default=Path("results/sonicom_complete_ten_method_test_v1/per_subject_metrics.csv"),
    )
    parser.add_argument(
        "--lap-source",
        type=Path,
        default=Path("results/lap2024_test_metrics_v1/lap2024_test_per_subject_metrics.csv"),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output}")
    for source in (args.main_source, args.supplementary_source, args.lap_source):
        if not source.is_file():
            raise FileNotFoundError(source)

    main_rows = read_csv(args.main_source)
    supplementary_rows = read_csv(args.supplementary_source)
    lap_rows = read_csv(args.lap_source)

    main_map = require_unique_map(
        [
            row
            for row in main_rows
            if row["Metric"] in {"FullSphereERB", "Contralateral25ERB"}
        ],
        key_fields=("SubjectLabel", "Method", "Metric"),
        value_field="Value_dB",
        label="FullSphereERB",
    )
    supplementary_map = require_unique_map(
        [
            row
            for row in supplementary_rows
            if row["Endpoint"] in {"ERBBandILDMean", "ITDWeightedMAE_us"}
        ],
        key_fields=("SubjectLabel", "Method", "Endpoint"),
        value_field="Value",
        label="supplementary endpoint",
    )
    lap_map = require_unique_map(
        lap_rows,
        key_fields=("SubjectLabel", "Method"),
        value_field="LAP2024LSD_dB",
        label="LAP2024LSD_dB",
    )

    main_subjects = subject_set(row["SubjectLabel"] for row in main_rows)
    supplementary_subjects = subject_set(row["SubjectLabel"] for row in supplementary_rows)
    lap_subjects = subject_set(row["SubjectLabel"] for row in lap_rows)
    if not (len(main_subjects) == len(supplementary_subjects) == len(lap_subjects) == 44):
        raise ValueError(
            "Expected 44 listeners in each source: "
            f"main={len(main_subjects)}, supplementary={len(supplementary_subjects)}, "
            f"lap={len(lap_subjects)}"
        )
    if not (main_subjects == supplementary_subjects == lap_subjects):
        raise ValueError("Listener sets differ across the five metric sources")
    subjects = sorted(main_subjects)

    values: dict[tuple[str, str], dict[str, float]] = {}
    for spec in METRIC_SPECS:
        metric = spec["metric"]
        for method in (CANDIDATE, *BASELINES):
            if spec["source_kind"] == "metric_long":
                source_values = {
                    subject: main_map[(subject, method, metric)] for subject in subjects
                }
            elif spec["source_kind"] == "per_subject":
                source_values = {
                    subject: supplementary_map[(subject, method, metric)]
                    for subject in subjects
                }
            else:
                source_values = {subject: lap_map[(subject, method)] for subject in subjects}
            if len(source_values) != 44 or not np.all(np.isfinite(list(source_values.values()))):
                raise ValueError(f"Incomplete or non-finite {metric}/{method}")
            values[(metric, method)] = source_values

    args.output.mkdir(parents=True, exist_ok=False)
    paired_rows: list[dict[str, object]] = []
    for spec in METRIC_SPECS:
        metric = spec["metric"]
        for baseline in BASELINES:
            stats = paired_tail_statistics(
                values[(metric, CANDIDATE)],
                values[(metric, baseline)],
                replicates=REPLICATES,
                seed=SEED,
            )
            paired_rows.append(
                {
                    "CandidateMethod": CANDIDATE,
                    "CandidateLabel": METHOD_LABELS[CANDIDATE],
                    "BaselineMethod": baseline,
                    "BaselineLabel": METHOD_LABELS[baseline],
                    "Metric": metric,
                    "MetricLabel": spec["metric_label"],
                    "Unit": spec["unit"],
                    "SourceEvidenceTier": spec["evidence_tier"],
                    "DifferenceDefinition": "FSC - baseline; negative favors FSC (lower is better)",
                    "MeanPairedDifference": stats["MeanDifference"],
                    "Bootstrap95Lower": stats["Bootstrap95Lower"],
                    "Bootstrap95Upper": stats["Bootstrap95Upper"],
                    "SubjectCount": stats["SubjectCount"],
                    "BootstrapReplicates": REPLICATES,
                    "BootstrapSeed": SEED,
                    "CandidateWins": stats["Wins"],
                    "Ties": stats["Ties"],
                    "CandidateLosses": stats["Losses"],
                }
            )

    paired_fields = list(paired_rows[0])
    write_csv(args.output / "paired_bootstrap.csv", paired_rows, paired_fields)

    table_rows: list[dict[str, object]] = []
    for spec in METRIC_SPECS:
        row: dict[str, object] = {
            "Metric": spec["metric"],
            "MetricLabel": spec["metric_label"],
            "Unit": spec["unit"],
            "SourceEvidenceTier": spec["evidence_tier"],
            "SubjectCount": 44,
            "BootstrapReplicates": REPLICATES,
            "BootstrapSeed": SEED,
        }
        for baseline in BASELINES:
            result = next(
                item
                for item in paired_rows
                if item["Metric"] == spec["metric"]
                and item["BaselineMethod"] == baseline
            )
            prefix = f"FSC_minus_{METHOD_LABELS[baseline].replace('-', '_').replace(' ', '_')}"
            row[f"{prefix}_Mean"] = result["MeanPairedDifference"]
            row[f"{prefix}_95CI"] = (
                f"[{float(result['Bootstrap95Lower']):.12g}, "
                f"{float(result['Bootstrap95Upper']):.12g}]"
            )
            row[f"{prefix}_Lower"] = result["Bootstrap95Lower"]
            row[f"{prefix}_Upper"] = result["Bootstrap95Upper"]
        table_rows.append(row)
    table_fields = list(table_rows[0])
    write_csv(args.output / "paper_summary_table.csv", table_rows, table_fields)

    write_csv(
        args.output / "listener_set.csv",
        ({"SubjectLabel": subject} for subject in subjects),
        ["SubjectLabel"],
    )

    source_hashes = {
        str(path): sha256(path)
        for path in (args.main_source, args.supplementary_source, args.lap_source)
    }
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "candidate": CANDIDATE,
        "candidate_label": METHOD_LABELS[CANDIDATE],
        "baselines": list(BASELINES),
        "metrics": [spec["metric"] for spec in METRIC_SPECS],
        "metric_count": len(METRIC_SPECS),
        "pair_count": len(paired_rows),
        "subject_count": len(subjects),
        "listener_set_match": True,
        "all_finite": True,
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
        "bootstrap_implementation": "mcar.evaluation.secondary_metrics.paired_tail_statistics",
        "difference_definition": "FSC - baseline; negative favors FSC; all metrics lower-is-better",
        "test_subject_count_read_during_derivation": 0,
        "training_or_inference_performed": False,
        "evidence_tiers_preserved": {
            spec["metric"]: spec["evidence_tier"] for spec in METRIC_SPECS
        },
        "source_files": {
            "FullSphereERB": str(args.main_source),
            "ERBBandILDMean": str(args.supplementary_source),
            "ITDWeightedMAE_us": str(args.supplementary_source),
            "LAP2024LSD_dB": str(args.lap_source),
        },
        "source_sha256": source_hashes,
        "listener_labels": subjects,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# FSC/Hybrid E190 unified five-metric paired statistics

This directory is a result-level derivation from frozen test CSV files. It does
not retrain, re-infer, or read raw SOFA/HDF5 data.

## Definition

For each metric and baseline, the statistic is computed from the same 44
listener-level values:

`difference_subject = FSC/Hybrid E190_subject - baseline_subject`

The reported mean is the mean of those paired differences. The 95% interval is
the percentile interval of 10,000 bootstrap means, resampling listeners with
replacement. Bootstrap uses the project's
`mcar.evaluation.secondary_metrics.paired_tail_statistics` implementation
(NumPy `default_rng`, linear quantiles), with fixed seed `{SEED}`. All metrics
are lower-is-better, so negative values favor FSC.

## Metrics and evidence tiers

| Metric | Endpoint | Evidence tier |
|---|---|---|
| Full-sphere ERB | `FullSphereERB` | Primary frozen engineering test |
| Contralateral-25 ERB | `Contralateral25ERB` | Primary frozen engineering test |
| ERB-band ILD mean | `ERBBandILDMean` | Secondary test |
| ITD weighted MAE | `ITDWeightedMAE_us` | Deferred test |
| LAP 2024 LSD | `LAP2024LSD_dB` | LAP2024 locked-test descriptive evaluation |

The source evidence tiers are retained; no metric is relabeled as preregistered
Primary. `paired_bootstrap.csv` contains all 15 requested pair-by-metric rows,
and `paper_summary_table.csv` is the compact paper-facing table. The LAP source
is the formally completed `results/lap2024_test_metrics_v1/` result with
`LAP2024LSD_dB`, 10 methods, and the same 44 listeners.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
