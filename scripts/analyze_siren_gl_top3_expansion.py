"""Aggregate the three-seed Top-3 expansion and freeze the final configuration."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, median

from mcar.paths import project_root


PROTOCOL = (
    "experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_FINAL_TRAINING_FREEZE.md"
)
CANDIDATES = {
    "warmup_cosine_c0": [
        "sonicom_film_siren_gl_c3_warmup_cosine_seed20260821_e150",
        "sonicom_film_siren_gl_top3_rank1_gl_c3_warmup_cosine_seed20260822_e150",
        "sonicom_film_siren_gl_top3_rank1_gl_c3_warmup_cosine_seed20260823_e150",
    ],
    "constant_c0": [
        "sonicom_film_siren_gl_c2_adamw_wd1e4_seed20260821_e150",
        "sonicom_film_siren_gl_top3_rank2_gl_c2_adamw_wd1e4_seed20260822_e150",
        "sonicom_film_siren_gl_top3_rank2_gl_c2_adamw_wd1e4_seed20260823_e150",
    ],
    "cosine_c0": [
        "sonicom_film_siren_gl_c3_cosine_seed20260821_e150",
        "sonicom_film_siren_gl_top3_rank3_gl_c3_cosine_seed20260822_e150",
        "sonicom_film_siren_gl_top3_rank3_gl_c3_cosine_seed20260823_e150",
    ],
}


def config_identity(configuration: dict[str, object]) -> tuple[object, ...]:
    objective = configuration["objective"]
    return (
        configuration["optimizer"]["name"],
        configuration["optimizer"]["learning_rate"],
        configuration["optimizer"]["weight_decay"],
        configuration["scheduler"]["name"],
        configuration["scheduler"]["horizon_cycles"],
        configuration["scheduler"]["warmup_cycles"],
        objective["erb_weight"],
        objective["high_frequency_weight"],
        objective["strict_ild_weight"],
        objective["spectral_band_ild_weight"],
        objective["high_frequency_first_difference_weight"],
        objective["high_frequency_second_difference_weight"],
        objective["notch_depth_weight"],
    )


def load_run(root: Path, run_name: str) -> dict[str, object]:
    run_dir = root / "artifacts" / "training" / run_name
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
    ledger = json.loads((run_dir / "validation_ledger.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (root / "configs" / "experiments" / f"{run_name}.json").read_text(
            encoding="utf-8"
        )
    )
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    numeric_values = [
        float(value)
        for row in history
        for key, value in row.items()
        if key != "cycle" and value not in (None, "")
    ]
    best_cycle = int(report["best_cycle"])
    best_entries = [entry for entry in ledger if int(entry["cycle"]) == best_cycle]
    checks = {
        "status": report.get("status") == "completed",
        "cycles": int(report.get("cycles", -1)) == 150,
        "history_rows": len(history) == 150,
        "finite": all(math.isfinite(value) for value in numeric_values),
        "decision": report.get("decision") == "KEEP",
        "best_not_last": best_cycle != 150,
        "test_not_read": int(report.get("test_subjects_read", -1)) == 0,
        "git_clean": report["git"]["dirty"] is False,
        "best_ledger_entry": len(best_entries) == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{run_name}: failed checks {failed}")
    metrics = best_entries[0]["aggregate"]["objective_metrics"]
    total = float(report["best_validation_objective_total"])
    if not math.isclose(total, float(metrics["total"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{run_name}: report/ledger total mismatch")
    return {
        "run_name": run_name,
        "seed": int(report["seed"]),
        "best_cycle": best_cycle,
        "best_validation_objective_total": total,
        "strict_ild_mae_db": float(metrics["ild_mae_db"]),
        "erb_mae_db": float(metrics["erb_mae_db"]),
        "hf_mae_db": float(metrics["contralateral_high_frequency_mae_db"]),
        "residual_mae_db": float(metrics["residual_mae_db"]),
        "config_identity": config_identity(configuration),
        "test_subjects_read": 0,
    }


def main() -> None:
    root = project_root()
    ranking: list[dict[str, object]] = []
    for candidate_name, run_names in CANDIDATES.items():
        runs = [load_run(root, run_name) for run_name in run_names]
        if sorted(run["seed"] for run in runs) != [20260821, 20260822, 20260823]:
            raise ValueError(f"{candidate_name}: seed set mismatch")
        if len({run["config_identity"] for run in runs}) != 1:
            raise ValueError(f"{candidate_name}: configuration identity mismatch")
        best_cycles = [int(run["best_cycle"]) for run in runs]
        ranking.append(
            {
                "configuration": candidate_name,
                "run_names": run_names,
                "seeds": [int(run["seed"]) for run in runs],
                "best_cycles": best_cycles,
                "mean_validation_objective_total": fmean(
                    float(run["best_validation_objective_total"]) for run in runs
                ),
                "mean_strict_ild_mae_db": fmean(
                    float(run["strict_ild_mae_db"]) for run in runs
                ),
                "mean_erb_mae_db": fmean(float(run["erb_mae_db"]) for run in runs),
                "mean_hf_mae_db": fmean(float(run["hf_mae_db"]) for run in runs),
                "mean_residual_mae_db": fmean(
                    float(run["residual_mae_db"]) for run in runs
                ),
                "individual_runs": [
                    {key: value for key, value in run.items() if key != "config_identity"}
                    for run in runs
                ],
                "test_subjects_read": 0,
            }
        )
    ranking.sort(
        key=lambda row: (
            row["mean_validation_objective_total"],
            row["mean_strict_ild_mae_db"],
            row["mean_erb_mae_db"],
            row["mean_hf_mae_db"],
            row["mean_residual_mae_db"],
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    winner = ranking[0]
    e_final = round(median(winner["best_cycles"]))
    result = {
        "protocol": PROTOCOL,
        "selection_metric": "three_seed_mean_best_validation_objective_total",
        "tie_break_order": ["strict_ild", "erb", "hf", "residual_mae"],
        "ranking": ranking,
        "winner": winner["configuration"],
        "winner_source_config": (
            "configs/experiments/sonicom_film_siren_gl_c3_cosine_seed20260821_e150.json"
        ),
        "winner_best_cycles": winner["best_cycles"],
        "e_final": e_final,
        "formal_seeds": [20260821, 20260822, 20260823],
        "scheduler_horizon_cycles": 150,
        "checkpoint_policy": "fixed_stop_cycle_last",
        "ensemble": {"type": "residual_db_mean", "weights": [1 / 3] * 3},
        "test_subjects_read": 0,
    }
    if result["winner"] != "cosine_c0" or e_final != 140:
        raise ValueError(f"unexpected final selection: {result['winner']} E{e_final}")
    output = root / "results" / "sonicom_film_siren_gl_top3" / "final_selection.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
