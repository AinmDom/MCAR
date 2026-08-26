"""Summarize the GL-C4 objective revalidation and write its decision.json.

Four from-scratch runs share seed 20260821 and the GL-C3 winner base
(AdamW lr=1e-4 wd=1e-4, warmup_cosine warmup 5, 150 cycles):

  * c0          -- pure Stage C total (re-run of the inherited C0 objective)
  * d1d2        -- C0 + D1/D2 weights 0.25/0.15, >= 4 kHz
  * notch       -- C0 + multi-scale notch weight 0.30
  * d1d2_notch  -- C0 + D1/D2 + notch

Ranking uses ONLY these four new GL-C4 runs. The protocol tie-break order is
total -> strict ILD -> ERB -> HF -> residual MAE; any run whose best cycle is
150 is marked RETEST and is not convergence evidence (but still participates
in the ranking numbers).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root

PROTOCOL = (
    "experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_REVALIDATION_PROTOCOL.md"
)
AMENDMENT = (
    "experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_C4_PARALLEL_AMENDMENT.md"
)
RUNS = {
    "c0": "sonicom_film_siren_gl_c4_c0_seed20260821_e150",
    "d1d2": "sonicom_film_siren_gl_c4_d1d2_seed20260821_e150",
    "notch": "sonicom_film_siren_gl_c4_notch_seed20260821_e150",
    "d1d2_notch": "sonicom_film_siren_gl_c4_d1d2_notch_seed20260821_e150",
}
EXPECTED_OBJECTIVE_WEIGHTS = {
    "c0": (0.0, 0.0, 0.0),
    "d1d2": (0.25, 0.15, 0.0),
    "notch": (0.0, 0.0, 0.30),
    "d1d2_notch": (0.25, 0.15, 0.30),
}
EXPECTED_HISTORY_ROWS = 150


def load_run(root: Path, key: str) -> dict[str, object]:
    name = RUNS[key]
    run_dir = root / "artifacts" / "training" / name
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    configuration = provenance["configuration"]
    ledger = json.loads(
        (run_dir / "validation_ledger.json").read_text(encoding="utf-8")
    )

    best_cycle = int(report["best_cycle"])
    best_entry = next(
        (entry for entry in ledger if int(entry["cycle"]) == best_cycle), None
    )
    if best_entry is None:
        raise ValueError(f"{name}: best cycle {best_cycle} missing from ledger")

    per_subject = best_entry["aggregate"]["per_subject"]
    objective_metrics = best_entry["aggregate"]["objective_metrics"]

    history_path = run_dir / "history.csv"
    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    data_rows = lines[1:]
    history_rows = len(data_rows)
    history_finite = True
    for row in data_rows:
        for value in row.split(","):
            if value.strip() == "":
                continue
            if not np.isfinite(float(value)):
                history_finite = False
                break
        if not history_finite:
            break

    expected_weights = EXPECTED_OBJECTIVE_WEIGHTS[key]
    objective = configuration["objective"]
    actual_weights = (
        float(objective["high_frequency_first_difference_weight"]),
        float(objective["high_frequency_second_difference_weight"]),
        float(objective["notch_depth_weight"]),
    )
    weights_match = np.allclose(actual_weights, expected_weights, atol=1e-9)

    checks = {
        "status": report.get("status") == "completed",
        "run_name": report.get("run_name") == name,
        "config_run_name": configuration.get("run_name") == name,
        "seed": int(report.get("seed", -1)) == 20260821,
        "config_seed": int(configuration.get("seed", -1)) == 20260821,
        "cycles": int(report.get("cycles", -1)) == 150,
        "scheduler": configuration["scheduler"].get("name") == "warmup_cosine"
        and configuration["scheduler"].get("warmup_cycles") == 5,
        "optimizer": configuration["optimizer"].get("name") == "AdamW"
        and np.isclose(
            float(configuration["optimizer"].get("learning_rate", -1.0)), 1e-4
        )
        and np.isclose(
            float(configuration["optimizer"].get("weight_decay", -1.0)), 1e-4
        ),
        "coordinate_dimension": configuration["model"]["coordinate_dimension"] == 7,
        "scope": configuration.get("conditioning_scope") == "global_plus_local_mca",
        "objective_weights": weights_match,
        "finite": bool(np.isfinite(float(report["best_validation_objective_total"])))
        and bool(np.all(np.isfinite([float(x["objective_total"]) for x in per_subject]))),
        "ledger_total_matches": np.isclose(
            float(objective_metrics["total"]),
            float(report["best_validation_objective_total"]),
            rtol=0.0,
            atol=1e-10,
        ),
        "history_rows": history_rows == EXPECTED_HISTORY_ROWS,
        "history_finite": history_finite,
        "git_clean": report["git"]["dirty"] is False,
        "test_not_read": int(report.get("test_subjects_read", -1)) == 0,
        "local_mca_reads": int(report.get("local_mca_inputs_read", -1)) == 40620,
    }
    failed = [key_ for key_, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{name}: failed checks {failed}")

    return {
        "key": key,
        "run_name": name,
        "best_cycle": best_cycle,
        "best_validation_objective_total": float(
            report["best_validation_objective_total"]
        ),
        "training_decision": str(report["decision"]),
        "strict_ild_mae_db": float(
            np.mean([float(x["strict_ild_mae_db"]) for x in per_subject])
        ),
        "erb_mae_db": float(
            np.mean([float(x["erb_mae_db"]) for x in per_subject])
        ),
        "hf_mae_db": float(
            np.mean(
                [float(x["contralateral_high_frequency_mae_db"]) for x in per_subject]
            )
        ),
        "residual_mae_db": float(
            np.mean([float(x["weighted_residual_mae_db"]) for x in per_subject])
        ),
        "test_subjects_read": 0,
    }


def sort_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    return (
        float(row["best_validation_objective_total"]),
        float(row["strict_ild_mae_db"]),
        float(row["erb_mae_db"]),
        float(row["hf_mae_db"]),
        float(row["residual_mae_db"]),
    )


def main() -> None:
    root = project_root()
    missing = [
        RUNS[key]
        for key in RUNS
        if not (root / "artifacts" / "training" / RUNS[key] / "training_report.json").is_file()
    ]
    if missing:
        raise SystemExit(
            "missing training reports (wait for all GL-C4 runs to finish):\n  "
            + "\n  ".join(missing)
        )
    rows = [load_run(root, key) for key in RUNS]
    rows.sort(key=sort_key)

    candidates = []
    for row in rows:
        decision = "RETEST" if int(row["best_cycle"]) == 150 else "KEEP"
        candidates.append(
            {
                "objective_variant": row["key"],
                "run_name": row["run_name"],
                "best_cycle": row["best_cycle"],
                "best_validation_objective_total": row[
                    "best_validation_objective_total"
                ],
                "best_cycle_strict_ild_mae_db": row["strict_ild_mae_db"],
                "best_cycle_erb_mae_db": row["erb_mae_db"],
                "best_cycle_hf_mae_db": row["hf_mae_db"],
                "best_cycle_residual_mae_db": row["residual_mae_db"],
                "decision": decision,
                "test_subjects_read": row["test_subjects_read"],
            }
        )

    ranking = [entry["objective_variant"] for entry in candidates]
    winner_variant = candidates[0]["objective_variant"]
    has_retest = any(entry["decision"] == "RETEST" for entry in candidates)
    winner = (
        None
        if has_retest
        else {
            "objective_variant": winner_variant,
            "run_name": candidates[0]["run_name"],
            "best_cycle": candidates[0]["best_cycle"],
            "best_validation_objective_total": candidates[0][
                "best_validation_objective_total"
            ],
            "decision": "KEEP",
        }
    )

    result = {
        "protocol": PROTOCOL,
        "execution_amendment": AMENDMENT,
        "search_stage": "global_local_c4_objective",
        "seed": 20260821,
        "cycles": 150,
        "selection_metric": "mean_stage_c_objective_total",
        "candidates": candidates,
        "ranking": ranking,
        "winner": winner,
        "next_stage": "GL-C4 winner feeds the global Top-3 selection across all 13 screening runs; the Top 3 each receive seeds 20260822/20260823.",
        "test_subjects_read": 0,
    }

    output_dir = root / "results" / "sonicom_film_siren_gl_c4_objective"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
