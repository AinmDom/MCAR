"""Aggregate the four Stage A4 confirmation runs and apply the frozen criteria.

Reads ``artifacts/training/sonicom_siren_a4_c{1..4}/matrix_summary.json`` and
writes:

- ``artifacts/siren_a4_confirmation/summary.csv`` (aggregate + per-subject);
- ``artifacts/siren_a4_confirmation/decision.json`` (frozen criteria check);
- copies under ``results/sonicom_siren_a4_confirmation/``.

Criteria (STAGE_A4_CONFIRMATION_PROTOCOL.md section 5):
M = c1 (main), L = c2 (historical baseline), D = c3 (A2 control),
A = c4 (depth tight control).

- main: M <= L
- secondary: M <= D
- plausibility: 2.4 <= M <= 3.4
- informational: |M - A| relative gap; paired statistics on 32 subjects.
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from pathlib import Path

from scipy import stats as scipy_stats

from mcar.paths import project_root

CELLS = {
    "M": ("sonicom_siren_a4_c1", "main candidate D-o20-d6-w256-ho20"),
    "L": ("sonicom_siren_a4_c2", "historical baseline linear-o30-d6-w256-ho30"),
    "D": ("sonicom_siren_a4_c3", "A2 control D-o30-d6-w256-ho30"),
    "A": ("sonicom_siren_a4_c4", "depth tight control D-o20-d4-w256-ho20"),
}
SUBJECT_LABELS = None  # resolved from reports


def load_cell(training_root: Path, run_name: str) -> dict:
    path = training_root / run_name / "matrix_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def paired_t(values_a: list[float], values_b: list[float]) -> float:
    """Two-sided paired t-test p-value on differences (scipy)."""
    result = scipy_stats.ttest_rel(values_a, values_b)
    return float(result.pvalue)


def main() -> None:
    root = project_root()
    training_root = root / "artifacts" / "training"
    artifact_dir = root / "artifacts" / "siren_a4_confirmation"
    result_dir = root / "results" / "sonicom_siren_a4_confirmation"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict] = {}
    for key, (run_name, _desc) in CELLS.items():
        cells[key] = load_cell(training_root, run_name)

    # per-subject holdout RMSE from the main cell's subject order
    labels = [sub["subject_label"] for sub in cells["M"]["subjects"]]
    per_subject: dict[str, dict[str, float]] = {}
    for key, cell in cells.items():
        by_label = {sub["subject_label"]: sub for sub in cell["subjects"]}
        if set(by_label) != set(labels):
            raise ValueError(f"cell {key} subject labels differ")
        per_subject[key] = {
            label: by_label[label]["best_holdout_metrics"]["residual_rmse_db"]
            for label in labels
        }

    def agg(key: str) -> float:
        return float(cells[key]["aggregate_holdout_rmse_db"])

    M, L, D, A = agg("M"), agg("L"), agg("D"), agg("A")
    checks = {
        "main_M_le_L": M <= L,
        "secondary_M_le_D": M <= D,
        "plausibility_2_4_le_M_le_3_4": 2.4 <= M <= 3.4,
    }
    gap_MA = 100.0 * (A - M) / M
    passed = all(checks.values())
    decision = {
        "criteria": "STAGE_A4_CONFIRMATION_PROTOCOL.md section 5",
        "M_main_candidate": M,
        "L_historical_baseline": L,
        "D_a2_control": D,
        "A_depth_tight_control": A,
        "checks": checks,
        "info_gap_M_vs_A_percent": round(gap_MA, 4),
        "paired": {
            "M_minus_L_mean_db": round(statistics.mean(
                [per_subject["M"][lab] - per_subject["L"][lab] for lab in labels]), 4),
            "M_minus_L_p": round(paired_t(
                [per_subject["M"][lab] for lab in labels],
                [per_subject["L"][lab] for lab in labels]), 6),
            "M_minus_D_mean_db": round(statistics.mean(
                [per_subject["M"][lab] - per_subject["D"][lab] for lab in labels]), 4),
            "M_minus_D_p": round(paired_t(
                [per_subject["M"][lab] for lab in labels],
                [per_subject["D"][lab] for lab in labels]), 6),
            "M_minus_A_mean_db": round(statistics.mean(
                [per_subject["M"][lab] - per_subject["A"][lab] for lab in labels]), 4),
            "M_minus_A_p": round(paired_t(
                [per_subject["M"][lab] for lab in labels],
                [per_subject["A"][lab] for lab in labels]), 6),
        },
        "passed": passed,
    }
    decision_path = artifact_dir / "decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for key in ("M", "L", "D", "A"):
        cell = cells[key]
        row = {
            "cell": key,
            "run_name": cell["run_name"],
            "description": dict(CELLS)[key][1],
            "aggregate_holdout_rmse_db": round(agg(key), 6),
            "aggregate_holdout_mae_db": round(
                float(cell["aggregate_holdout_mae_db"]), 6),
            "aggregate_full_field_rmse_db": round(
                float(cell["aggregate_full_field_rmse_db"]), 6),
        }
        for label in labels:
            row[f"holdout_rmse_{label}_db"] = round(per_subject[key][label], 6)
        rows.append(row)
    csv_path = artifact_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for name in ("summary.csv", "decision.json"):
        shutil.copy2(artifact_dir / name, result_dir / name)

    print("=== Stage A4 confirmation (32-subject holdout RMSE, lower is better) ===")
    for row in rows:
        print(f"{row['cell']}: {row['aggregate_holdout_rmse_db']:.4f} dB  ({row['description']})")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
