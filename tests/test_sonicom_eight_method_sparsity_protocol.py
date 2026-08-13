from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_eight_method_protocol_is_nested_and_matches_learned_study() -> None:
    config = json.loads(
        (ROOT / "configs/experiments/sonicom_eight_method_sparsity_v1.json").read_text()
    )
    learned = json.loads(
        (ROOT / "configs/experiments/sonicom_learned_methods_sparsity_v1.json").read_text()
    )
    assert config["dataset"]["direction_counts"] == [6, 14, 26]
    assert config["dataset"]["subject_count"] == 44
    assert config["dataset"]["sparse_grid_file"] == learned["dataset"]["sparse_grid_file"]
    assert config["dataset"]["reference_grid"] == learned["dataset"]["reference_grid"]
    assert len(config["baseline_methods"]) == 5
    assert config["existing_learned_result"]["methods"] == [
        "MCARv32",
        "FSPAE",
        "RANF",
    ]


def test_existing_learned_table_is_complete_and_unique() -> None:
    path = ROOT / "results/sonicom_learned_methods_sparsity_v1/metric_long.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 44 * 3 * 3 * 4
    keys = {
        (
            row["SubjectID"],
            row["DirectionCount"],
            row["Method"],
            row["Metric"],
        )
        for row in rows
    }
    assert len(keys) == len(rows)
    assert {row["Method"] for row in rows} == {"MCARv32", "FSPAE", "RANF"}


def test_combined_analysis_declares_all_eight_methods() -> None:
    from mcar.evaluation.analyze_sonicom_eight_method_sparsity import (
        METHOD_IDS,
        METRICS,
    )

    assert METHOD_IDS == (
        "SHOnly",
        "SUpDEqSH",
        "SUpDEqNN",
        "SUpDEqBary",
        "MCA",
        "MCARv32",
        "FSPAE",
        "RANF",
    )
    assert len(METRICS) == 4


def test_five_baseline_result_is_complete_and_finite() -> None:
    path = ROOT / "results/sonicom_five_baseline_sparsity_v1/metric_long.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 44 * 3 * 5 * 4
    assert len(
        {
            (row["SubjectID"], row["DirectionCount"], row["Method"], row["Metric"])
            for row in rows
        }
    ) == len(rows)
    assert all(math.isfinite(float(row["Value_dB"])) for row in rows)


def test_combined_result_is_complete_and_reuses_learned_rows_exactly() -> None:
    combined_path = ROOT / "results/sonicom_eight_method_sparsity_v1/metric_long.csv"
    learned_path = ROOT / "results/sonicom_learned_methods_sparsity_v1/metric_long.csv"
    with combined_path.open(newline="", encoding="utf-8-sig") as stream:
        combined = list(csv.DictReader(stream))
    with learned_path.open(newline="", encoding="utf-8-sig") as stream:
        learned = list(csv.DictReader(stream))
    assert len(combined) == 44 * 3 * 8 * 4
    learned_values = {
        (row["SubjectID"], row["DirectionCount"], row["Method"], row["Metric"]): float(
            row["Value_dB"]
        )
        for row in learned
    }
    combined_learned = {
        (row["SubjectID"], row["DirectionCount"], row["Method"], row["Metric"]): float(
            row["Value_dB"]
        )
        for row in combined
        if row["Method"] in {"MCARv32", "FSPAE", "RANF"}
    }
    assert combined_learned == learned_values


def test_combined_inference_tables_have_preregistered_sizes() -> None:
    root = ROOT / "results/sonicom_eight_method_sparsity_v1"
    expected = {
        "aggregate_metrics.csv": 8 * 3 * 4,
        "robustness_summary.csv": 8 * 4,
        "paired_bootstrap_all_pairs.csv": 28 * 4 * 2,
        "paired_bootstrap_learned_vs_baselines.csv": 15 * 4 * 2,
        "rankings.csv": 4 * 4 * 8,
    }
    for name, row_count in expected.items():
        with (root / name).open(newline="", encoding="utf-8-sig") as stream:
            assert len(list(csv.DictReader(stream))) == row_count
