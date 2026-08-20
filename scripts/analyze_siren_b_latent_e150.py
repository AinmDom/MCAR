"""Audit the uniform E150 Stage B latent reruns and freeze their winner."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcar.paths import project_root


LATENT_DIMENSIONS = (128, 256)
SEEDS = (20260821, 20260822, 20260823)
EARLY_WINDOW = (100, 105, 110, 115, 120, 125)
LATE_WINDOW = (130, 135, 140, 145, 150)
MINIMUM_WINDOW_IMPROVEMENT_PERCENT = 0.1


def run_name(latent_dimension: int, seed: int) -> str:
    return f"sonicom_film_siren_b_latent_{latent_dimension}_seed{seed}_e150"


def plateau_budget_decision(
    best_cycle: int,
    validation_by_cycle: dict[int, float],
) -> dict[str, float | str]:
    """Apply the preregistered E150 plateau rule to one completed run."""
    if best_cycle < 150:
        return {
            "budget_status": "KEEP",
            "early_window_mean_mae_db": float("nan"),
            "late_window_mean_mae_db": float("nan"),
            "window_improvement_percent": float("nan"),
        }
    if best_cycle != 150:
        raise ValueError(f"Expected best_cycle <= 150, found {best_cycle}")
    missing = sorted(set(EARLY_WINDOW + LATE_WINDOW) - validation_by_cycle.keys())
    if missing:
        raise ValueError(f"Missing validation cycles for plateau audit: {missing}")
    early = float(np.mean([validation_by_cycle[cycle] for cycle in EARLY_WINDOW]))
    late = float(np.mean([validation_by_cycle[cycle] for cycle in LATE_WINDOW]))
    improvement = 100.0 * (early - late) / early
    status = (
        "RETEST"
        if improvement + 1e-12 >= MINIMUM_WINDOW_IMPROVEMENT_PERCENT
        else "KEEP_PLATEAU"
    )
    return {
        "budget_status": status,
        "early_window_mean_mae_db": early,
        "late_window_mean_mae_db": late,
        "window_improvement_percent": improvement,
    }


def read_validation_history(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["cycle"]): float(row["validation_weighted_mae_db"])
            for row in rows
            if row["validation_weighted_mae_db"]
        }


def read_run(root: Path, latent_dimension: int, seed: int) -> dict[str, object] | None:
    name = run_name(latent_dimension, seed)
    directory = root / "artifacts" / "training" / name
    report_path = directory / "training_report.json"
    subject_path = directory / "best_validation_per_subject.csv"
    history_path = directory / "history.csv"
    if not report_path.is_file():
        return None
    if not subject_path.is_file() or not history_path.is_file():
        raise ValueError(f"Incomplete artifacts for {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "completed" or report["cycles"] != 150:
        raise ValueError(f"Incomplete or wrong-budget report for {name}")
    if report["run_name"] != name or report["seed"] != seed:
        raise ValueError(f"Run metadata mismatch for {name}")
    if report["test_subjects_read"] != 0:
        raise ValueError(f"{name} read test subjects")

    with subject_path.open("r", encoding="utf-8", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    if len(subjects) != 44:
        raise ValueError(f"Expected 44 validation rows for {name}, found {len(subjects)}")
    mae = np.asarray([float(row["weighted_mae_db"]) for row in subjects])
    rmse = np.asarray([float(row["weighted_rmse_db"]) for row in subjects])
    baseline = np.asarray(
        [float(row["mca_zero_residual_weighted_mae_db"]) for row in subjects]
    )
    aggregate_mae = float(np.mean(mae))
    if not np.isclose(
        aggregate_mae,
        float(report["best_validation_weighted_mae_db"]),
        atol=1e-10,
    ):
        raise ValueError(f"Aggregate mismatch for {name}")

    plateau = plateau_budget_decision(
        int(report["best_cycle"]),
        read_validation_history(history_path),
    )
    return {
        "latent_dimension": latent_dimension,
        "seed": seed,
        "run_name": name,
        "best_cycle": int(report["best_cycle"]),
        "weighted_mae_db": aggregate_mae,
        "weighted_rmse_db": float(np.mean(rmse)),
        "median_mae_db": float(np.median(mae)),
        "std_subject_mae_db": float(np.std(mae, ddof=0)),
        "mca_zero_residual_mae_db": float(np.mean(baseline)),
        "improvement_vs_mca_percent": float(
            100.0 * (1.0 - aggregate_mae / np.mean(baseline))
        ),
        **plateau,
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "peak_cuda_allocated_mib": float(report["peak_cuda_allocated_mib"]),
        "checkpoint_sha256": str(report["best_checkpoint_sha256"]),
        "git_commit": str(report["git"]["commit"]),
        "git_dirty": bool(report["git"]["dirty"]),
        "test_subjects_read": 0,
    }


def analyze(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        row
        for latent_dimension in LATENT_DIMENSIONS
        for seed in SEEDS
        if (row := read_run(root, latent_dimension, seed)) is not None
    ]
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["latent_dimension"])].append(row)
    complete = all(len(grouped[latent]) == len(SEEDS) for latent in LATENT_DIMENSIONS)
    retest = any(row["budget_status"] == "RETEST" for row in rows)
    ranking: list[dict[str, float | int]] = []
    if complete:
        for latent in LATENT_DIMENSIONS:
            values = np.asarray(
                [float(row["weighted_mae_db"]) for row in grouped[latent]]
            )
            ranking.append(
                {
                    "latent_dimension": latent,
                    "mean_weighted_mae_db": float(np.mean(values)),
                    "std_seed_mae_db": float(np.std(values, ddof=0)),
                    "minimum_weighted_mae_db": float(np.min(values)),
                    "maximum_weighted_mae_db": float(np.max(values)),
                }
            )
        ranking.sort(key=lambda item: float(item["mean_weighted_mae_db"]))
    decision: dict[str, object] = {
        "protocol": "experiments/film_siren/STAGE_B_LATENT_BUDGET_AMENDMENT.md",
        "plateau_rule": "experiments/film_siren/STAGE_B_LATENT_PLATEAU_AMENDMENT.md",
        "cycles": 150,
        "expected_runs": len(LATENT_DIMENSIONS) * len(SEEDS),
        "completed_runs": len(rows),
        "three_seed_complete": complete,
        "budget_retest_required": retest,
        "three_seed_ranking": ranking,
        "winner": (
            int(ranking[0]["latent_dimension"])
            if complete and not retest
            else None
        ),
        "test_subjects_read": 0,
    }
    rows.sort(key=lambda row: (int(row["latent_dimension"]), int(row["seed"])))
    return rows, decision


def main() -> None:
    root = project_root()
    rows, decision = analyze(root)
    output_dir = root / "results" / "sonicom_film_siren_b_latent"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "e150_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "e150_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
