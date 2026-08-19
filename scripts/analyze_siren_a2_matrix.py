"""Aggregate the nine Stage A2 matrix runs and apply the frozen Top-2 rule.

Reads ``artifacts/training/sonicom_siren_a2_<mode>_w256_d6_o<omega>/matrix_summary.json``
for the nine frozen configurations and writes:

- ``artifacts/siren_a2_matrix/summary.csv``: per-config aggregate and
  per-subject holdout RMSE rows;
- ``artifacts/siren_a2_matrix/top2.json``: the Top-2 decision with the
  frozen rule rationale from STAGE_A2_PROTOCOL.md section 6.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mcar.paths import project_root

MODES = ("linear", "erb", "dual")
OMEGAS = (20, 30, 50)
FREQUENCY_LABELS = {"linear": "L", "erb": "E", "dual": "D"}


def load_summaries(training_root: Path) -> list[dict]:
    summaries: list[dict] = []
    for mode in MODES:
        for omega in OMEGAS:
            run_name = f"sonicom_siren_a2_{mode}_w256_d6_o{omega}"
            path = training_root / run_name / "matrix_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def rank_configurations(
    summaries: list[dict],
) -> list[tuple[dict, float, float]]:
    ranked: list[tuple[dict, float, float]] = []
    for summary in summaries:
        holdout_rmse = float(summary["aggregate_holdout_rmse_db"])
        holdout_mae = float(summary["aggregate_holdout_mae_db"])
        ranked.append((summary, holdout_rmse, holdout_mae))
    ranked.sort(key=lambda item: item[1])
    return ranked


def top_two_decision(
    ranked: list[tuple[dict, float, float]],
) -> dict:
    first, second, third = ranked[0], ranked[1], ranked[2]
    first_rmse, second_rmse, third_rmse = first[1], second[1], third[1]
    gap_12 = 100.0 * (second_rmse - first_rmse) / first_rmse
    gap_13 = 100.0 * (third_rmse - first_rmse) / first_rmse
    tight_second = gap_12 < 0.5
    clear_third = gap_13 > 2.0
    first_name = (
        f"{FREQUENCY_LABELS[first[0]['frequency_mode']]}"
        f"-o{int(first[0]['first_omega'])}"
    )
    second_name = (
        f"{FREQUENCY_LABELS[second[0]['frequency_mode']]}"
        f"-o{int(second[0]['first_omega'])}"
    )
    if tight_second and clear_third:
        rationale = (
            f"First ({first_name}) and second ({second_name}) are within "
            f"{gap_12:.3f}% (<0.5%) while the third is {gap_13:.2f}% away "
            "(>2%); promote both into depth/width/hidden-omega search."
        )
    else:
        rationale = (
            f"First ({first_name}) leads by {gap_12:.3f}% over second "
            f"({second_name}); promote first as the primary configuration "
            f"and carry second as a comparison control."
        )
    return {
        "rule": "STAGE_A2_PROTOCOL.md section 6 Top-2 rule",
        "first": first_name,
        "second": second_name,
        "gap_first_second_percent": round(gap_12, 4),
        "gap_first_third_percent": round(gap_13, 4),
        "tight_second": tight_second,
        "clear_third": clear_third,
        "rationale": rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root",
        type=Path,
        default=None,
        help="artifacts/training root (default: project artifacts/training)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = project_root()
    training_root = (args.training_root or root / "artifacts" / "training").resolve()
    output_dir = (args.output_dir or root / "artifacts" / "siren_a2_matrix").resolve()
    summaries = load_summaries(training_root)
    ranked = rank_configurations(summaries)

    rows: list[dict] = []
    for rank, (summary, holdout_rmse, holdout_mae) in enumerate(ranked, start=1):
        full_rmse = float(summary["aggregate_full_field_rmse_db"])
        full_mae = float(summary["aggregate_full_field_mae_db"])
        row = {
            "rank": rank,
            "frequency_mode": summary["frequency_mode"],
            "first_omega": int(summary["first_omega"]),
            "run_name": summary["run_name"],
            "aggregate_holdout_rmse_db": round(holdout_rmse, 6),
            "aggregate_holdout_mae_db": round(holdout_mae, 6),
            "aggregate_full_field_rmse_db": round(full_rmse, 6),
            "aggregate_full_field_mae_db": round(full_mae, 6),
        }
        for subject in summary["subjects"]:
            row[f"holdout_rmse_{subject['subject_label']}_db"] = round(
                subject["best_holdout_metrics"]["residual_rmse_db"], 6
            )
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    decision = top_two_decision(ranked)
    decision["summary_csv"] = str(csv_path)
    decision["generated_from_run_count"] = len(ranked)
    decision_path = output_dir / "top2.json"
    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    print("=== Stage A2 matrix ranking (holdout RMSE, lower is better) ===")
    header = f"{'rank':>4} {'config':>8} {'holdoutRMSE':>12} {'holdoutMAE':>11} {'fullRMSE':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        label = (
            f"{FREQUENCY_LABELS[row['frequency_mode']]}-"
            f"o{int(row['first_omega'])}"
        )
        print(
            f"{row['rank']:>4} {label:>8} "
            f"{row['aggregate_holdout_rmse_db']:>12.4f} "
            f"{row['aggregate_holdout_mae_db']:>11.4f} "
            f"{row['aggregate_full_field_rmse_db']:>10.4f}"
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
