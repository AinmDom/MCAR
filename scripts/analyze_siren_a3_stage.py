"""Aggregate a Stage A3 search stage and apply the frozen Top decision rule.

Usage::

    python scripts/analyze_siren_a3_stage.py --stage depth \
        --runs sonicom_siren_a3_d4,sonicom_siren_a3_d6,sonicom_siren_a3_d8

Reads each run's ``matrix_summary.json`` under ``artifacts/training/``, ranks
by aggregate holdout RMSE, and writes:

- ``artifacts/siren_a3_matrix/<stage>/summary.csv``;
- ``artifacts/siren_a3_matrix/<stage>/top.json`` (frozen rule rationale);
- a copy of both under ``results/sonicom_siren_a3_matrix/<stage>/``.

The Top rule matches STAGE_A2_PROTOCOL.md section 6: promote the top two when
the second is within 0.5% and the third is more than 2% away, otherwise the
first is the primary configuration and the second the control.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from mcar.paths import project_root


def load_run_summary(training_root: Path, run_name: str) -> dict:
    path = training_root / run_name / "matrix_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--runs", required=True,
                        help="comma-separated run names for this stage")
    parser.add_argument("--labels", default=None,
                        help="comma-separated display labels, optional")
    args = parser.parse_args()

    root = project_root()
    training_root = root / "artifacts" / "training"
    run_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    if len(run_names) < 2:
        raise ValueError("At least two runs are required for a stage ranking")
    labels = (
        [label.strip() for label in args.labels.split(",")]
        if args.labels
        else run_names
    )
    if len(labels) != len(run_names):
        raise ValueError("labels must match run count")

    summaries = [load_run_summary(training_root, name) for name in run_names]
    ranked = sorted(
        zip(summaries, labels),
        key=lambda item: float(item[0]["aggregate_holdout_rmse_db"]),
    )

    rows: list[dict] = []
    for rank, (summary, label) in enumerate(ranked, start=1):
        row = {
            "rank": rank,
            "label": label,
            "run_name": summary["run_name"],
            "hidden_width": summary["hidden_width"],
            "sine_layer_count": summary["sine_layer_count"],
            "hidden_omega": summary["hidden_omega"],
            "aggregate_holdout_rmse_db": round(
                float(summary["aggregate_holdout_rmse_db"]), 6
            ),
            "aggregate_holdout_mae_db": round(
                float(summary["aggregate_holdout_mae_db"]), 6
            ),
            "aggregate_full_field_rmse_db": round(
                float(summary["aggregate_full_field_rmse_db"]), 6
            ),
        }
        for subject in summary["subjects"]:
            row[f"holdout_rmse_{subject['subject_label']}_db"] = round(
                subject["best_holdout_metrics"]["residual_rmse_db"], 6
            )
        rows.append(row)

    artifact_dir = root / "artifacts" / "siren_a3_matrix" / args.stage
    result_dir = root / "results" / "sonicom_siren_a3_matrix" / args.stage
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifact_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    first, second, third = ranked[0], ranked[1], ranked[2]
    gap_12 = 100.0 * (float(second[0]["aggregate_holdout_rmse_db"]) - float(first[0]["aggregate_holdout_rmse_db"])) / float(first[0]["aggregate_holdout_rmse_db"])
    gap_13 = 100.0 * (float(third[0]["aggregate_holdout_rmse_db"]) - float(first[0]["aggregate_holdout_rmse_db"])) / float(first[0]["aggregate_holdout_rmse_db"])
    tight_second = gap_12 < 0.5
    clear_third = gap_13 > 2.0
    if tight_second and clear_third:
        rationale = (
            f"First ({first[1]}) and second ({second[1]}) are within "
            f"{gap_12:.3f}% (<0.5%) while the third is {gap_13:.2f}% away "
            "(>2%); promote both into the next stage."
        )
    else:
        rationale = (
            f"First ({first[1]}) leads by {gap_12:.3f}% over second "
            f"({second[1]}); promote first as the primary configuration and "
            f"carry second as the comparison control."
        )
    decision = {
        "stage": args.stage,
        "rule": "STAGE_A2_PROTOCOL.md section 6 Top rule",
        "first": first[1],
        "second": second[1],
        "gap_first_second_percent": round(gap_12, 4),
        "gap_first_third_percent": round(gap_13, 4),
        "tight_second": tight_second,
        "clear_third": clear_third,
        "rationale": rationale,
    }
    decision_path = artifact_dir / "top.json"
    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("summary.csv", "top.json"):
        shutil.copy2(artifact_dir / name, result_dir / name)

    print(f"=== Stage A3 {args.stage} ranking (holdout RMSE, lower is better) ===")
    header = f"{'rank':>4} {'label':>18} {'holdRMSE':>10} {'holdMAE':>10} {'fullRMSE':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rank']:>4} {row['label']:>18} "
            f"{row['aggregate_holdout_rmse_db']:>10.4f} "
            f"{row['aggregate_holdout_mae_db']:>10.4f} "
            f"{row['aggregate_full_field_rmse_db']:>10.4f}"
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
