"""Select the global Top-3 configurations across all Stage C screening runs.

All 13 screening runs (GL-C1 x3, GL-C2 x3, GL-C3 x3, GL-C4 x4) use seed
20260821 and 150 cycles. Ranking follows the protocol tie-break order
total -> strict ILD -> ERB -> HF -> residual MAE, computed at each run's best
cycle from its validation ledger. RETEST runs (best cycle == 150) still
participate in the ranking numbers but are flagged.

Top-3 is selected per UNIQUE configuration (optimizer + scheduler + objective
identity), because GL-C4's c0 re-runs the GL-C3 winner configuration: within a
group only the best run ranks, and each of the three groups then receives the
seed expansion. Run only after all four GL-C4 runs have finished naturally:
the script fails if any of the 13 expected training_report.json files is
missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root

PROTOCOL = (
    "experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_REVALIDATION_PROTOCOL.md"
)
EXPECTED_RUNS = [
    # GL-C1 learning rate
    "sonicom_film_siren_gl_c1_lr3e5_adam_seed20260821_e150",
    "sonicom_film_siren_gl_c1_lr1e4_adam_seed20260821_e150",
    "sonicom_film_siren_gl_c1_lr3e4_adam_seed20260821_e150",
    # GL-C2 optimizer / weight decay
    "sonicom_film_siren_gl_c2_adam_wd0_seed20260821_e150",
    "sonicom_film_siren_gl_c2_adamw_wd1e5_seed20260821_e150",
    "sonicom_film_siren_gl_c2_adamw_wd1e4_seed20260821_e150",
    # GL-C3 scheduler
    "sonicom_film_siren_gl_c3_constant_seed20260821_e150",
    "sonicom_film_siren_gl_c3_cosine_seed20260821_e150",
    "sonicom_film_siren_gl_c3_warmup_cosine_seed20260821_e150",
    # GL-C4 objective
    "sonicom_film_siren_gl_c4_c0_seed20260821_e150",
    "sonicom_film_siren_gl_c4_d1d2_seed20260821_e150",
    "sonicom_film_siren_gl_c4_notch_seed20260821_e150",
    "sonicom_film_siren_gl_c4_d1d2_notch_seed20260821_e150",
]


def load_run(root: Path, name: str) -> dict[str, object]:
    run_dir = root / "artifacts" / "training" / name
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
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
    checks = {
        "status": report.get("status") == "completed",
        "run_name": report.get("run_name") == name,
        "cycles": int(report.get("cycles", -1)) == 150,
        "seed": int(report.get("seed", -1)) == 20260821,
        "finite": bool(
            np.all(np.isfinite([float(x["objective_total"]) for x in per_subject]))
        ),
        "git_clean": report["git"]["dirty"] is False,
        "test_not_read": int(report.get("test_subjects_read", -1)) == 0,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{name}: failed checks {failed}")
    return {
        "run_name": name,
        "best_cycle": best_cycle,
        "best_validation_objective_total": float(
            report["best_validation_objective_total"]
        ),
        "training_decision": str(report["decision"]),
        "strict_ild_mae_db": float(
            np.mean([float(x["strict_ild_mae_db"]) for x in per_subject])
        ),
        "erb_mae_db": float(np.mean([float(x["erb_mae_db"]) for x in per_subject])),
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
        name
        for name in EXPECTED_RUNS
        if not (root / "artifacts" / "training" / name / "training_report.json").is_file()
    ]
    if missing:
        raise SystemExit(
            "missing training reports (wait for all GL-C4 runs to finish):\n  "
            + "\n  ".join(missing)
        )

    rows = [load_run(root, name) for name in EXPECTED_RUNS]
    config_root = root / "configs" / "experiments"
    for row in rows:
        configuration = json.loads(
            (config_root / f"{row['run_name']}.json").read_text(encoding="utf-8")
        )
        row["config_identity"] = config_identity(configuration)

    rows.sort(key=sort_key)

    full_ranking = []
    for rank, row in enumerate(rows, start=1):
        entry = {
            "rank": rank,
            "run_name": row["run_name"],
            "best_cycle": row["best_cycle"],
            "best_validation_objective_total": row["best_validation_objective_total"],
            "strict_ild_mae_db": row["strict_ild_mae_db"],
            "erb_mae_db": row["erb_mae_db"],
            "hf_mae_db": row["hf_mae_db"],
            "residual_mae_db": row["residual_mae_db"],
            "decision": "RETEST" if int(row["best_cycle"]) == 150 else "KEEP",
            "test_subjects_read": row["test_subjects_read"],
        }
        full_ranking.append(entry)

    # Group by unique configuration, keep the best run per group.
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(row["config_identity"], []).append(row)
    unique_ranking = []
    for identity, members in groups.items():
        best = min(members, key=sort_key)
        unique_ranking.append(
            {
                "run_name": best["run_name"],
                "member_run_names": sorted(m["run_name"] for m in members),
                "member_count": len(members),
                "best_cycle": best["best_cycle"],
                "best_validation_objective_total": best[
                    "best_validation_objective_total"
                ],
                "decision": "RETEST" if int(best["best_cycle"]) == 150 else "KEEP",
                "test_subjects_read": 0,
            }
        )
    unique_ranking.sort(
        key=lambda entry: float(entry["best_validation_objective_total"])
    )

    top3 = [
        {"rank": rank, "run_name": entry["run_name"]}
        for rank, entry in enumerate(unique_ranking[:3], start=1)
    ]

    result = {
        "protocol": PROTOCOL,
        "search_stage": "global_top3_selection_across_screening",
        "seed": 20260821,
        "cycles": 150,
        "selection_metric": "mean_stage_c_objective_total",
        "screening_run_count": len(EXPECTED_RUNS),
        "unique_configuration_count": len(unique_ranking),
        "full_ranking": full_ranking,
        "unique_ranking": unique_ranking,
        "top3": top3,
        "top3_note": (
            "Top-3 is selected per unique configuration (optimizer+scheduler+"
            "objective); GL-C4 c0 re-runs the GL-C3 winner configuration, so "
            "the two share one slot."
        ),
        "test_subjects_read": 0,
    }

    output_dir = root / "results" / "sonicom_film_siren_gl_top3"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
