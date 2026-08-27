"""Aggregate the three D1/D2+notch development seeds and freeze E_final."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, median

from mcar.paths import project_root


AMENDMENT = (
    "experiments/film_siren/"
    "STAGE_C_GLOBAL_LOCAL_MCA_C4_COMMON_SCORE_CORRECTION.md"
)
RUNS = {
    20260821: "sonicom_film_siren_gl_c4_d1d2_notch_seed20260821_e150",
    20260822: (
        "sonicom_film_siren_gl_c4_corrected_d1d2_notch_seed20260822_e150"
    ),
    20260823: (
        "sonicom_film_siren_gl_c4_corrected_d1d2_notch_seed20260823_e150"
    ),
}


def standardized_c0(metrics: dict[str, object], target_std: float) -> float:
    return float(metrics["residual_smooth_l1"]) + (
        0.75 * float(metrics["erb_mae_db"])
        + 0.25 * float(metrics["contralateral_high_frequency_mae_db"])
        + 0.75 * float(metrics["ild_mae_db"])
        + 0.05 * float(metrics["spectral_band_ild_smooth_l1_db"])
    ) / target_std


def load_run(root: Path, seed: int, run_name: str, target_std: float) -> dict[str, object]:
    run_root = root / "artifacts" / "training" / run_name
    report = json.loads(
        (run_root / "training_report.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (run_root / "validation_ledger.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (run_root / "configuration.json").read_text(encoding="utf-8")
    )["configuration"]
    with (run_root / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    numeric = [
        float(value)
        for row in history
        for key, value in row.items()
        if key != "cycle" and value not in (None, "")
    ]
    best_cycle = int(report["best_cycle"])
    best_rows = [row for row in ledger if int(row["cycle"]) == best_cycle]
    checks = {
        "status": report.get("status") == "completed",
        "run_name": report.get("run_name") == run_name,
        "seed": int(report.get("seed", -1)) == seed,
        "cycles": int(report.get("cycles", -1)) == 150,
        "history_rows": len(history) == 150,
        "finite": all(math.isfinite(value) for value in numeric),
        "best_not_last": best_cycle != 150,
        "decision": report.get("decision") == "KEEP",
        "test_not_read": int(report.get("test_subjects_read", -1)) == 0,
        "git_clean": report["git"]["dirty"] is False,
        "best_ledger_row": len(best_rows) == 1,
        "scope": provenance.get("conditioning_scope") == "global_plus_local_mca",
        "objective": (
            provenance["objective"]["high_frequency_first_difference_weight"],
            provenance["objective"]["high_frequency_second_difference_weight"],
            provenance["objective"]["notch_depth_weight"],
        )
        == (0.25, 0.15, 0.30),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{run_name}: failed checks {failed}")
    metrics = best_rows[0]["aggregate"]["objective_metrics"]
    raw_total = float(report["best_validation_objective_total"])
    if not math.isclose(raw_total, float(metrics["total"]), abs_tol=1e-12):
        raise ValueError(f"{run_name}: report/ledger total mismatch")
    return {
        "run_name": run_name,
        "seed": seed,
        "best_cycle": best_cycle,
        "augmented_objective_total": raw_total,
        "standardized_c0": standardized_c0(metrics, target_std),
        "residual_mae_db": float(metrics["residual_mae_db"]),
        "erb_mae_db": float(metrics["erb_mae_db"]),
        "contralateral_high_frequency_mae_db": float(
            metrics["contralateral_high_frequency_mae_db"]
        ),
        "strict_ild_mae_db": float(metrics["ild_mae_db"]),
        "test_subjects_read": 0,
    }


def main() -> None:
    root = project_root()
    statistics = json.loads(
        (
            root
            / "data/processed/sonicom_residual_q26_v1/training_statistics.json"
        ).read_text(encoding="utf-8")
    )
    target_std = float(
        statistics["per_tensor"]["target_residual_db"]["std_population"]
    )
    runs = [load_run(root, seed, name, target_std) for seed, name in RUNS.items()]
    best_cycles = [int(run["best_cycle"]) for run in runs]
    e_final = round(median(best_cycles))
    result = {
        "schema_version": "1.0",
        "protocol_correction": AMENDMENT,
        "candidate": "warmup_cosine_d1d2_notch",
        "within_candidate_checkpoint_metric": (
            "augmented_d1d2_notch_objective_total"
        ),
        "cross_objective_diagnostic_metric": "standardized_c0",
        "target_std_train_only": target_std,
        "runs": runs,
        "three_seed_mean": {
            "augmented_objective_total": fmean(
                float(run["augmented_objective_total"]) for run in runs
            ),
            "standardized_c0": fmean(float(run["standardized_c0"]) for run in runs),
            "residual_mae_db": fmean(float(run["residual_mae_db"]) for run in runs),
            "erb_mae_db": fmean(float(run["erb_mae_db"]) for run in runs),
            "contralateral_high_frequency_mae_db": fmean(
                float(run["contralateral_high_frequency_mae_db"]) for run in runs
            ),
            "strict_ild_mae_db": fmean(
                float(run["strict_ild_mae_db"]) for run in runs
            ),
        },
        "best_cycles": best_cycles,
        "e_final": e_final,
        "formal_seeds": list(RUNS),
        "scheduler_horizon_cycles": 150,
        "formal_checkpoint_policy": "fixed_stop_cycle_last",
        "ensemble": {"type": "residual_db_mean", "weights": [1 / 3] * 3},
        "test_subjects_read": 0,
    }
    output = (
        root
        / "results/sonicom_film_siren_gl_c4_corrected_expansion/selection.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
