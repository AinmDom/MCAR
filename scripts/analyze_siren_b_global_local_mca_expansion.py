"""Analyze the frozen three-model, three-seed global/local-MCA expansion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root


SCOPES = ("global_zero_local", "local_mca", "global_plus_local_mca")
SEEDS = (20260821, 20260822, 20260823)
GATE_RUNS = {
    "global_zero_local": "sonicom_film_siren_b_gate_global_zero_local_seed20260821_e150",
    "global_plus_local_mca": "sonicom_film_siren_b_gate_global_plus_local_mca_seed20260821_e150",
}
GATE_PROTOCOL = "experiments/film_siren/STAGE_B_GLOBAL_LOCAL_MCA_GATE_PROTOCOL.md"
EXPANSION_PROTOCOL = "experiments/film_siren/STAGE_B_GLOBAL_LOCAL_MCA_EXPANSION_PROTOCOL.md"


def run_name(scope: str, seed: int) -> str:
    if seed == 20260821 and scope in GATE_RUNS:
        return GATE_RUNS[scope]
    return f"sonicom_film_siren_b_expansion_{scope}_seed{seed}_e150"


def load_run(root: Path, scope: str, seed: int) -> dict[str, object]:
    name = run_name(scope, seed)
    run_dir = root / "artifacts" / "training" / name
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    configuration = provenance["configuration"]
    with (run_dir / "best_validation_per_subject.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        subjects = list(csv.DictReader(handle))
    mae = np.asarray([float(row["weighted_mae_db"]) for row in subjects])
    aggregate = float(np.mean(mae))
    expected_local_reads = 0 if scope == "global_zero_local" else 40620
    expected_protocol = (
        GATE_PROTOCOL if seed == 20260821 and scope in GATE_RUNS else EXPANSION_PROTOCOL
    )
    checks = {
        "44_subjects": len(subjects) == 44,
        "status": report.get("status") == "completed",
        "run_name": report.get("run_name") == name,
        "scope": configuration.get("conditioning_scope") == scope,
        "report_scope": report.get("conditioning_scope") == scope,
        "protocol": configuration.get("protocol_amendment") == expected_protocol,
        "seed": int(report.get("seed", -1)) == seed,
        "cycles": int(report.get("cycles", -1)) == 150,
        "coordinate_dimension": configuration["model"]["coordinate_dimension"] == 7,
        "architecture": (
            configuration["model"]["latent_dimension"] == 128
            and configuration["model"]["modulation_variant"] == "full"
            and configuration["model"]["placement"] == "all"
        ),
        "finite": bool(np.all(np.isfinite(mae))) and np.isfinite(aggregate),
        "aggregate": np.isclose(
            aggregate,
            float(report["best_validation_weighted_mae_db"]),
            rtol=0.0,
            atol=1e-10,
        ),
        "git_clean": report["git"]["dirty"] is False,
        "local_reads": int(report.get("local_mca_inputs_read", -1))
        == expected_local_reads,
        "test_not_read": int(report.get("test_subjects_read", -1)) == 0,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{name}: failed checks {failed}")
    return {
        "conditioning_scope": scope,
        "seed": seed,
        "run_name": name,
        "best_cycle": int(report["best_cycle"]),
        "weighted_mae_db": aggregate,
        "training_decision": str(report["decision"]),
        "local_mca_inputs_read": expected_local_reads,
        "test_subjects_read": 0,
    }


def main() -> None:
    root = project_root()
    rows = [load_run(root, scope, seed) for scope in SCOPES for seed in SEEDS]
    ranking = []
    for scope in SCOPES:
        values = np.asarray(
            [float(row["weighted_mae_db"]) for row in rows if row["conditioning_scope"] == scope]
        )
        ranking.append(
            {
                "conditioning_scope": scope,
                "mean_weighted_mae_db": float(np.mean(values)),
                "std_weighted_mae_db": float(np.std(values, ddof=0)),
                "minimum_weighted_mae_db": float(np.min(values)),
                "maximum_weighted_mae_db": float(np.max(values)),
            }
        )
    ranking.sort(key=lambda row: float(row["mean_weighted_mae_db"]))
    has_retest = any(row["training_decision"] == "RETEST" for row in rows)
    decision = "RETEST" if has_retest else "FREEZE_WINNER"
    winner = None if has_retest else str(ranking[0]["conditioning_scope"])
    seed_winners = {
        str(seed): min(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: float(row["weighted_mae_db"]),
        )["conditioning_scope"]
        for seed in SEEDS
    }
    output_dir = root / "results" / "sonicom_film_siren_b_global_local_mca_expansion"
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking[0].keys())
        writer.writeheader()
        writer.writerows(ranking)
    result = {
        "status": "blocked" if has_retest else "complete",
        "decision": decision,
        "winner": winner,
        "ranking": ranking,
        "seed_winners": seed_winners,
        "protocol": EXPANSION_PROTOCOL,
        "test_subjects_read": 0,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
