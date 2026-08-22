"""Compare the frozen conditioned winner with shared-SIREN baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root


SEEDS = (20260821, 20260822, 20260823)
PROTOCOL = "experiments/film_siren/STAGE_B_UNCONDITIONAL_BASELINE_PROTOCOL.md"


def run_name(seed: int) -> str:
    return f"sonicom_shared_siren_b_unconditional_seed{seed}"


def read_conditioned_run(root: Path, placement: str, seed: int) -> dict[str, object] | None:
    name = f"sonicom_film_siren_b_placement_{placement}_seed{seed}"
    directory = root / "artifacts" / "training" / name
    report_path = directory / "training_report.json"
    subject_path = directory / "best_validation_per_subject.csv"
    if not report_path.is_file():
        return None
    if not subject_path.is_file():
        raise ValueError(f"Missing per-subject validation results for {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("cycles") != 100:
        raise ValueError(f"Incomplete or wrong-budget report for {name}")
    if report.get("run_name") != name or report.get("seed") != seed:
        raise ValueError(f"Run metadata mismatch for {name}")
    if report.get("test_subjects_read") != 0 or bool(report.get("git", {}).get("dirty")):
        raise ValueError(f"Unsafe data/Git state for {name}")
    with subject_path.open("r", encoding="utf-8", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    if len(subjects) != 44:
        raise ValueError(f"Expected 44 validation rows for {name}")
    mae = np.asarray([float(row["weighted_mae_db"]) for row in subjects])
    if not np.isfinite(mae).all():
        raise ValueError(f"Non-finite result for {name}")
    aggregate = float(np.mean(mae))
    if not np.isclose(aggregate, report["best_validation_weighted_mae_db"], atol=1e-10):
        raise ValueError(f"Aggregate mismatch for {name}")
    return {
        "seed": seed,
        "run_name": name,
        "weighted_mae_db": aggregate,
        "decision": str(report["decision"]),
        "test_subjects_read": 0,
    }


def read_baseline_run(root: Path, seed: int) -> dict[str, object] | None:
    name = run_name(seed)
    directory = root / "artifacts" / "training" / name
    report_path = directory / "training_report.json"
    subject_path = directory / "best_validation_per_subject.csv"
    if not report_path.is_file():
        return None
    if not subject_path.is_file():
        raise ValueError(f"Missing per-subject validation results for {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("cycles") != 100:
        raise ValueError(f"Incomplete or wrong-budget report for {name}")
    if report.get("run_name") != name or report.get("seed") != seed:
        raise ValueError(f"Run metadata mismatch for {name}")
    if (
        report.get("condition_inputs_read") != 0
        or report.get("test_subjects_read") != 0
        or bool(report.get("git", {}).get("dirty"))
    ):
        raise ValueError(f"Unsafe condition/data/Git state for {name}")
    with subject_path.open("r", encoding="utf-8", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    if len(subjects) != 44:
        raise ValueError(f"Expected 44 validation rows for {name}")
    mae = np.asarray([float(row["weighted_mae_db"]) for row in subjects])
    rmse = np.asarray([float(row["weighted_rmse_db"]) for row in subjects])
    baseline = np.asarray(
        [float(row["mca_zero_residual_weighted_mae_db"]) for row in subjects]
    )
    if not (np.isfinite(mae).all() and np.isfinite(rmse).all() and np.isfinite(baseline).all()):
        raise ValueError(f"Non-finite result for {name}")
    aggregate = float(np.mean(mae))
    if not np.isclose(aggregate, report["best_validation_weighted_mae_db"], atol=1e-10):
        raise ValueError(f"Aggregate mismatch for {name}")
    return {
        "seed": seed,
        "run_name": name,
        "best_cycle": int(report["best_cycle"]),
        "weighted_mae_db": aggregate,
        "weighted_rmse_db": float(np.mean(rmse)),
        "median_mae_db": float(np.median(mae)),
        "std_subject_mae_db": float(np.std(mae, ddof=0)),
        "mca_zero_residual_mae_db": float(np.mean(baseline)),
        "decision": str(report["decision"]),
        "condition_inputs_read": 0,
        "test_subjects_read": 0,
        "git_commit": str(report["git"]["commit"]),
        "git_dirty": False,
    }


def conditioning_passes(
    conditioned_mean: float,
    unconditional_mean: float,
    *,
    complete: bool,
    budget_retest: bool,
) -> bool | None:
    if not complete or budget_retest:
        return None
    if not (np.isfinite(conditioned_mean) and np.isfinite(unconditional_mean)):
        return None
    return bool(conditioned_mean < unconditional_mean)


def analyze(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_rows = [
        row for seed in SEEDS if (row := read_baseline_run(root, seed)) is not None
    ]
    conditioned_rows = [read_conditioned_run(root, "all", seed) for seed in SEEDS]
    conditioned_complete = all(row is not None for row in conditioned_rows)
    complete = len(baseline_rows) == len(SEEDS) and conditioned_complete
    budget_retest = any(row["decision"] == "RETEST" for row in baseline_rows)
    baseline_values = np.asarray(
        [float(row["weighted_mae_db"]) for row in baseline_rows]
    )
    conditioned_values = np.asarray(
        [float(row["weighted_mae_db"]) for row in conditioned_rows if row is not None]
    )
    baseline_mean = float(np.mean(baseline_values)) if len(baseline_values) == 3 else None
    conditioned_mean = (
        float(np.mean(conditioned_values)) if len(conditioned_values) == 3 else None
    )
    passed = (
        conditioning_passes(
            conditioned_mean,
            baseline_mean,
            complete=complete,
            budget_retest=budget_retest,
        )
        if conditioned_mean is not None and baseline_mean is not None
        else None
    )
    paired = []
    if complete:
        baseline_by_seed = {int(row["seed"]): float(row["weighted_mae_db"]) for row in baseline_rows}
        conditioned_by_seed = {
            int(row["seed"]): float(row["weighted_mae_db"])
            for row in conditioned_rows
            if row is not None
        }
        paired = [
            {
                "seed": seed,
                "conditioned_minus_unconditional_mae_db": conditioned_by_seed[seed]
                - baseline_by_seed[seed],
            }
            for seed in SEEDS
        ]
    decision: dict[str, object] = {
        "protocol": PROTOCOL,
        "conditioned_architecture": "latent128_full_all",
        "seeds": list(SEEDS),
        "completed_baseline_runs": len(baseline_rows),
        "expected_baseline_runs": 3,
        "comparison_complete": complete,
        "budget_retest_required": budget_retest,
        "conditioned_mean_weighted_mae_db": conditioned_mean,
        "conditioned_std_seed_mae_db": (
            float(np.std(conditioned_values, ddof=0)) if complete else None
        ),
        "unconditional_mean_weighted_mae_db": baseline_mean,
        "unconditional_std_seed_mae_db": (
            float(np.std(baseline_values, ddof=0)) if complete else None
        ),
        "conditioned_relative_improvement_percent": (
            float(100.0 * (1.0 - conditioned_mean / baseline_mean))
            if complete and baseline_mean != 0.0
            else None
        ),
        "paired_deltas": paired,
        "conditioned_seed_wins": (
            sum(float(item["conditioned_minus_unconditional_mae_db"]) < 0.0 for item in paired)
            if complete
            else None
        ),
        "conditioning_check_passed": passed,
        "stage_b_status": (
            "CONDITIONING_CHECK_PASSED"
            if passed is True
            else "DO_NOT_FREEZE"
            if passed is False
            else "PENDING"
        ),
        "condition_inputs_read_by_baseline": 0,
        "test_subjects_read": 0,
    }
    baseline_rows.sort(key=lambda row: int(row["seed"]))
    return baseline_rows, decision


def main() -> None:
    root = project_root()
    rows, decision = analyze(root)
    output_dir = root / "results" / "sonicom_shared_siren_b_unconditional"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
