"""Analyze the pre-registered Stage B global/local-MCA bounded gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root


RUNS = {
    "global_zero_local": "sonicom_film_siren_b_gate_global_zero_local_seed20260821_e150",
    "global_plus_local_mca": "sonicom_film_siren_b_gate_global_plus_local_mca_seed20260821_e150",
}
PROTOCOL = "experiments/film_siren/STAGE_B_GLOBAL_LOCAL_MCA_GATE_PROTOCOL.md"
THRESHOLD_PERCENT = 0.5


def load_run(root: Path, scope: str, run_name: str) -> dict[str, object]:
    run_dir = root / "artifacts" / "training" / run_name
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    configuration = provenance["configuration"]
    with (run_dir / "best_validation_per_subject.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 44:
        raise ValueError(f"{run_name}: expected 44 validation subjects")
    mae = np.asarray([float(row["weighted_mae_db"]) for row in rows])
    aggregate = float(np.mean(mae))
    expected_local_reads = 0 if scope == "global_zero_local" else 40620
    checks = {
        "status": report.get("status") == "completed",
        "run_name": report.get("run_name") == run_name,
        "scope": configuration.get("conditioning_scope") == scope,
        "report_scope": report.get("conditioning_scope") == scope,
        "protocol": configuration.get("protocol_amendment") == PROTOCOL,
        "seed": int(report.get("seed", -1)) == 20260821,
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
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{run_name}: failed checks {failed}")
    return {
        "conditioning_scope": scope,
        "run_name": run_name,
        "best_cycle": int(report["best_cycle"]),
        "weighted_mae_db": aggregate,
        "training_decision": str(report["decision"]),
        "local_mca_inputs_read": expected_local_reads,
        "test_subjects_read": 0,
    }


def main() -> None:
    root = project_root()
    rows = [load_run(root, scope, run_name) for scope, run_name in RUNS.items()]
    by_scope = {str(row["conditioning_scope"]): row for row in rows}
    global_mae = float(by_scope["global_zero_local"]["weighted_mae_db"])
    combined_mae = float(by_scope["global_plus_local_mca"]["weighted_mae_db"])
    improvement = (global_mae - combined_mae) / global_mae * 100.0
    has_retest = any(row["training_decision"] == "RETEST" for row in rows)
    decision = (
        "RETEST"
        if has_retest
        else "PASS_EXPAND"
        if improvement >= THRESHOLD_PERCENT
        else "KEEP_GLOBAL"
    )
    output_dir = root / "results" / "sonicom_film_siren_b_global_local_mca_gate"
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "complete" if decision != "RETEST" else "blocked",
        "decision": decision,
        "winner": (
            "global_plus_local_mca"
            if decision == "PASS_EXPAND"
            else "global_zero_local"
            if decision == "KEEP_GLOBAL"
            else None
        ),
        "threshold_percent": THRESHOLD_PERCENT,
        "global_weighted_mae_db": global_mae,
        "global_plus_local_mca_weighted_mae_db": combined_mae,
        "relative_improvement_percent": improvement,
        "protocol": PROTOCOL,
        "test_subjects_read": 0,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
