"""Aggregate the frozen five-seed Stage B placement RETEST."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from mcar.paths import project_root


PLACEMENTS = ("hidden", "all")
ORIGINAL_SEEDS = (20260821, 20260822, 20260823)
RETEST_SEEDS = (20260824, 20260825)
SEEDS = ORIGINAL_SEEDS + RETEST_SEEDS
PROTOCOL = "experiments/film_siren/STAGE_B_PLACEMENT_RETEST_PROTOCOL.md"


def run_name(placement: str, seed: int) -> str:
    suffix = "_retest" if seed in RETEST_SEEDS else ""
    return f"sonicom_film_siren_b_placement_{placement}_seed{seed}{suffix}"


def read_run(root: Path, placement: str, seed: int) -> dict[str, object] | None:
    name = run_name(placement, seed)
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
        raise ValueError(f"Unsafe data/Git state recorded for {name}")
    with subject_path.open("r", encoding="utf-8", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    if len(subjects) != 44:
        raise ValueError(f"Expected 44 validation rows for {name}, found {len(subjects)}")
    mae = np.asarray([float(row["weighted_mae_db"]) for row in subjects])
    rmse = np.asarray([float(row["weighted_rmse_db"]) for row in subjects])
    baseline = np.asarray(
        [float(row["mca_zero_residual_weighted_mae_db"]) for row in subjects]
    )
    if not (np.isfinite(mae).all() and np.isfinite(rmse).all() and np.isfinite(baseline).all()):
        raise ValueError(f"Non-finite validation result for {name}")
    aggregate_mae = float(np.mean(mae))
    if not np.isclose(
        aggregate_mae,
        float(report["best_validation_weighted_mae_db"]),
        atol=1e-10,
    ):
        raise ValueError(f"Aggregate mismatch for {name}")
    return {
        "placement": placement,
        "seed": seed,
        "run_name": name,
        "is_retest_run": seed in RETEST_SEEDS,
        "best_cycle": int(report["best_cycle"]),
        "weighted_mae_db": aggregate_mae,
        "weighted_rmse_db": float(np.mean(rmse)),
        "median_mae_db": float(np.median(mae)),
        "std_subject_mae_db": float(np.std(mae, ddof=0)),
        "mca_zero_residual_mae_db": float(np.mean(baseline)),
        "decision": str(report["decision"]),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "peak_cuda_allocated_mib": float(report["peak_cuda_allocated_mib"]),
        "checkpoint_sha256": str(report["best_checkpoint_sha256"]),
        "git_commit": str(report["git"]["commit"]),
        "git_dirty": False,
        "test_subjects_read": 0,
    }


def select_retest_winner(
    means: dict[str, float], *, complete: bool, budget_retest: bool
) -> str | None:
    if not complete or budget_retest or set(means) != set(PLACEMENTS):
        return None
    hidden = float(means["hidden"])
    all_layers = float(means["all"])
    if not (np.isfinite(hidden) and np.isfinite(all_layers)) or hidden == all_layers:
        return None
    return "hidden" if hidden < all_layers else "all"


def exact_paired_bootstrap_interval(deltas: np.ndarray) -> tuple[float, float]:
    """Return the exact 95% bootstrap interval for five paired seed deltas."""
    if deltas.shape != (5,) or not np.isfinite(deltas).all():
        raise ValueError("Expected five finite paired seed deltas")
    bootstrap_means = np.asarray(
        [float(np.mean(deltas[list(indices)])) for indices in product(range(5), repeat=5)]
    )
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def analyze(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        row
        for placement in PLACEMENTS
        for seed in SEEDS
        if (row := read_run(root, placement, seed)) is not None
    ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["placement"])].append(row)
    complete = all(
        {int(row["seed"]) for row in grouped[placement]} == set(SEEDS)
        for placement in PLACEMENTS
    )
    budget_retest = any(row["decision"] == "RETEST" for row in rows)
    ranking: list[dict[str, float | str]] = []
    means: dict[str, float] = {}
    if complete:
        for placement in PLACEMENTS:
            values = np.asarray(
                [float(row["weighted_mae_db"]) for row in grouped[placement]]
            )
            means[placement] = float(np.mean(values))
            ranking.append(
                {
                    "placement": placement,
                    "mean_weighted_mae_db": float(np.mean(values)),
                    "std_seed_mae_db": float(np.std(values, ddof=0)),
                    "minimum_weighted_mae_db": float(np.min(values)),
                    "maximum_weighted_mae_db": float(np.max(values)),
                }
            )
        ranking.sort(key=lambda item: float(item["mean_weighted_mae_db"]))
    paired: list[dict[str, float | int]] = []
    if complete:
        by_key = {
            (str(row["placement"]), int(row["seed"])): float(row["weighted_mae_db"])
            for row in rows
        }
        paired = [
            {
                "seed": seed,
                "all_minus_hidden_mae_db": by_key[("all", seed)]
                - by_key[("hidden", seed)],
            }
            for seed in SEEDS
        ]
    deltas = np.asarray([float(item["all_minus_hidden_mae_db"]) for item in paired])
    bootstrap_interval = exact_paired_bootstrap_interval(deltas) if complete else None
    winner = select_retest_winner(
        means, complete=complete, budget_retest=budget_retest
    )
    decision: dict[str, object] = {
        "protocol": PROTOCOL,
        "latent_dimension": 128,
        "modulation_variant": "full",
        "placements": list(PLACEMENTS),
        "seeds": list(SEEDS),
        "completed_runs": len(rows),
        "expected_runs": len(PLACEMENTS) * len(SEEDS),
        "five_seed_complete": complete,
        "budget_retest_required": budget_retest,
        "five_seed_ranking": ranking,
        "paired_deltas": paired,
        "all_seed_wins": int(np.sum(deltas < 0.0)) if complete else None,
        "hidden_seed_wins": int(np.sum(deltas > 0.0)) if complete else None,
        "paired_ties": int(np.sum(deltas == 0.0)) if complete else None,
        "mean_all_minus_hidden_mae_db": float(np.mean(deltas)) if complete else None,
        "paired_mean_bootstrap_95_interval_db": (
            list(bootstrap_interval) if bootstrap_interval is not None else None
        ),
        "winner": winner,
        "test_subjects_read": 0,
    }
    rows.sort(key=lambda row: (str(row["placement"]), int(row["seed"])))
    return rows, decision


def main() -> None:
    root = project_root()
    rows, decision = analyze(root)
    output_dir = root / "results" / "sonicom_film_siren_b_placement"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "retest_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "retest_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
