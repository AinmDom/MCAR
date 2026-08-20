"""Aggregate Stage B latent-dimension screening and three-seed follow-up."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcar.paths import project_root


LATENT_DIMENSIONS = (64, 128, 256)
SEEDS = (20260821, 20260822, 20260823)


def run_name(latent_dimension: int, seed: int) -> str:
    return f"sonicom_film_siren_b_latent_{latent_dimension}_seed{seed}"


def read_run(root: Path, latent_dimension: int, seed: int) -> dict[str, object] | None:
    directory = root / "artifacts" / "training" / run_name(latent_dimension, seed)
    report_path = directory / "training_report.json"
    subject_path = directory / "best_validation_per_subject.csv"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["test_subjects_read"] != 0:
        raise ValueError(f"{run_name(latent_dimension, seed)} read test subjects")
    if report["seed"] != seed or report["cycles"] != 100:
        raise ValueError(f"Run metadata mismatch for {run_name(latent_dimension, seed)}")
    with subject_path.open("r", encoding="utf-8", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    if len(subjects) != 44:
        raise ValueError(f"Expected 44 validation rows, found {len(subjects)}")
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
        raise ValueError(f"Aggregate mismatch for {run_name(latent_dimension, seed)}")
    return {
        "latent_dimension": latent_dimension,
        "seed": seed,
        "run_name": run_name(latent_dimension, seed),
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
        "git_dirty": bool(report["git"]["dirty"]),
        "test_subjects_read": 0,
    }


def main() -> None:
    root = project_root()
    rows = [
        row
        for latent_dimension in LATENT_DIMENSIONS
        for seed in SEEDS
        if (row := read_run(root, latent_dimension, seed)) is not None
    ]
    screening = [row for row in rows if row["seed"] == SEEDS[0]]
    if len(screening) != 3:
        raise ValueError("All three screening runs must exist")
    screening.sort(key=lambda row: float(row["weighted_mae_db"]))
    top2 = [int(row["latent_dimension"]) for row in screening[:2]]
    best = float(screening[0]["weighted_mae_db"])
    gaps = {
        str(row["latent_dimension"]): 100.0
        * (float(row["weighted_mae_db"]) - best)
        / best
        for row in screening
    }
    extend = top2[0] == 256 and gaps[str(top2[1])] < 1.0

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["latent_dimension"])].append(row)
    top2_complete = all(len(grouped[latent]) == 3 for latent in top2)
    three_seed: list[dict[str, float | int]] = []
    if top2_complete:
        for latent in top2:
            values = np.asarray(
                [float(row["weighted_mae_db"]) for row in grouped[latent]]
            )
            three_seed.append(
                {
                    "latent_dimension": latent,
                    "mean_weighted_mae_db": float(np.mean(values)),
                    "std_seed_mae_db": float(np.std(values, ddof=0)),
                    "minimum_weighted_mae_db": float(np.min(values)),
                    "maximum_weighted_mae_db": float(np.max(values)),
                }
            )
        three_seed.sort(key=lambda row: float(row["mean_weighted_mae_db"]))

    output_dir = root / "results" / "sonicom_film_siren_b_latent"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows.sort(key=lambda row: (int(row["latent_dimension"]), int(row["seed"])))
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    decision = {
        "protocol": "experiments/film_siren/STAGE_B_FILM_PROTOCOL.md section 8",
        "screening_seed": SEEDS[0],
        "screening_ranking": [
            {
                "rank": rank,
                "latent_dimension": int(row["latent_dimension"]),
                "weighted_mae_db": float(row["weighted_mae_db"]),
                "gap_from_best_percent": gaps[str(row["latent_dimension"])],
            }
            for rank, row in enumerate(screening, start=1)
        ],
        "screening_top2": top2,
        "screening_tight": gaps[str(top2[1])] < 0.5,
        "extend_384_512": extend,
        "three_seed_complete": top2_complete,
        "three_seed_ranking": three_seed,
        "winner": (
            int(three_seed[0]["latent_dimension"])
            if three_seed
            else None
        ),
        "test_subjects_read": 0,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
