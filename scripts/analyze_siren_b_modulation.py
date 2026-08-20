"""Aggregate Stage B modulation screening and Top-2 seed follow-up."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcar.paths import project_root


VARIANTS = ("concat", "amplitude", "phase", "full")
SEEDS = (20260821, 20260822, 20260823)


def run_name(variant: str, seed: int) -> str:
    return f"sonicom_film_siren_b_modulation_{variant}_seed{seed}"


def read_run(root: Path, variant: str, seed: int) -> dict[str, object] | None:
    name = run_name(variant, seed)
    directory = root / "artifacts" / "training" / name
    report_path = directory / "training_report.json"
    subject_path = directory / "best_validation_per_subject.csv"
    if not report_path.is_file():
        return None
    if not subject_path.is_file():
        raise ValueError(f"Missing per-subject validation results for {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "completed" or report["cycles"] != 100:
        raise ValueError(f"Incomplete or wrong-budget report for {name}")
    if report["run_name"] != name or report["seed"] != seed:
        raise ValueError(f"Run metadata mismatch for {name}")
    if report["test_subjects_read"] != 0:
        raise ValueError(f"{name} read test subjects")
    if bool(report["git"]["dirty"]):
        raise ValueError(f"{name} was trained from a dirty worktree")
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
    return {
        "variant": variant,
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
        "decision": str(report["decision"]),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "peak_cuda_allocated_mib": float(report["peak_cuda_allocated_mib"]),
        "checkpoint_sha256": str(report["best_checkpoint_sha256"]),
        "git_commit": str(report["git"]["commit"]),
        "git_dirty": False,
        "test_subjects_read": 0,
    }


def analyze(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        row
        for variant in VARIANTS
        for seed in SEEDS
        if (row := read_run(root, variant, seed)) is not None
    ]
    screening = [row for row in rows if row["seed"] == SEEDS[0]]
    screening.sort(key=lambda row: float(row["weighted_mae_db"]))
    screening_complete = len(screening) == len(VARIANTS)
    top2 = [str(row["variant"]) for row in screening[:2]] if screening_complete else []
    screening_ranking = [
        {
            "rank": rank,
            "variant": str(row["variant"]),
            "weighted_mae_db": float(row["weighted_mae_db"]),
        }
        for rank, row in enumerate(screening, start=1)
    ]
    screening_tight = False
    if screening_complete:
        first = float(screening[0]["weighted_mae_db"])
        second = float(screening[1]["weighted_mae_db"])
        screening_tight = 100.0 * (second - first) / first < 0.5

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    top2_complete = bool(top2) and all(len(grouped[name]) == 3 for name in top2)
    ranking: list[dict[str, float | str]] = []
    if top2_complete:
        for variant in top2:
            values = np.asarray(
                [float(row["weighted_mae_db"]) for row in grouped[variant]]
            )
            ranking.append(
                {
                    "variant": variant,
                    "mean_weighted_mae_db": float(np.mean(values)),
                    "std_seed_mae_db": float(np.std(values, ddof=0)),
                    "minimum_weighted_mae_db": float(np.min(values)),
                    "maximum_weighted_mae_db": float(np.max(values)),
                }
            )
        ranking.sort(key=lambda item: float(item["mean_weighted_mae_db"]))
    budget_retest = any(row["decision"] == "RETEST" for row in rows)
    ranking_reversed = bool(ranking) and str(ranking[0]["variant"]) != top2[0]
    retest = budget_retest or ranking_reversed
    decision: dict[str, object] = {
        "protocol": "experiments/film_siren/STAGE_B_FILM_PROTOCOL.md section 8",
        "latent_dimension": 128,
        "screening_seed": SEEDS[0],
        "screening_complete": screening_complete,
        "screening_ranking": screening_ranking,
        "screening_top2": top2,
        "screening_tight": screening_tight,
        "three_seed_complete": top2_complete,
        "budget_retest_required": budget_retest,
        "seed_ranking_reversed": ranking_reversed,
        "three_seed_ranking": ranking,
        "winner": (
            str(ranking[0]["variant"])
            if top2_complete and not retest
            else None
        ),
        "skip_placement": (
            bool(ranking) and str(ranking[0]["variant"]) == "concat" and not retest
        ),
        "test_subjects_read": 0,
    }
    rows.sort(key=lambda row: (str(row["variant"]), int(row["seed"])))
    return rows, decision


def main() -> None:
    root = project_root()
    rows, decision = analyze(root)
    output_dir = root / "results" / "sonicom_film_siren_b_modulation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
