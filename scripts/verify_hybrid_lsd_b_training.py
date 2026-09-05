"""Verify every fixed-cycle Hybrid LSD-B run before ensemble inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260821, 20260822, 20260823)
OUTPUT = ROOT / "reports/HYBRID_LSD_B_TRAINING_VERIFICATION_20260905.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    launch = json.loads((ROOT / "outputs/hybrid_lsd_b_e190/attempt2/launch.json").read_text())
    if launch["status"] != "processes_exited_successfully_pending_artifact_verification":
        raise ValueError("Controller did not record three successful exits")
    members = []
    for seed in SEEDS:
        entry = next(item for item in launch["members"] if item["seed"] == seed)
        if entry["exit_code"] != 0:
            raise ValueError(f"Nonzero exit for {seed}")
        run = ROOT / entry["output_directory"]
        report = json.loads((run / "training_report.json").read_text())
        configuration = json.loads((run / "configuration.json").read_text())
        history = list(csv.DictReader((run / "history.csv").open(newline="")))
        ledger = json.loads((run / "validation_ledger.json").read_text())
        numeric = [float(value) for row in history for value in row.values() if value]
        numeric.extend(numbers(ledger))
        if not all(math.isfinite(value) for value in numeric):
            raise FloatingPointError(f"Nonfinite history/ledger for {seed}")
        expected_cycles = list(range(5, 191, 5))
        checks = {
            "completed": report["status"] == "completed",
            "fixed_cycle_decision": report["decision"] == "FIXED_CYCLE_COMPLETE",
            "cycles_190": report["cycles"] == 190 and len(history) == 190,
            "optimizer_steps_49780": report["optimizer_steps"] == 49780 and int(history[-1]["optimizer_steps_completed"]) == 49780,
            "last_authoritative": report["authoritative_checkpoint"] == "last.pt" and report["checkpoint_policy"] == "fixed_stop_cycle_last",
            "validation_cycles_complete": [item["cycle"] for item in ledger] == expected_cycles,
            "all_history_ledger_numeric_finite": True,
            "test_subjects_read_zero": report["test_subjects_read"] == 0 and all(item["test_subjects_read"] == 0 for item in ledger),
            "clean_training_git": not report["git"]["dirty"] and not configuration["git"]["dirty"],
            "training_git_frozen": report["git"]["commit"] == "bd00151f7fc717fc748454e5419fc16d3f65d06e",
            "film_siren_frozen": report["film_siren_frozen"] is True,
            "lsd_weight_one": configuration["configuration"]["objective"]["lsd_weight"] == 1.0,
            "split_counts": configuration["train_subject_count"] == 262 and configuration["validation_subject_count"] == 44,
        }
        if not all(checks.values()):
            raise AssertionError({seed: checks})
        last_path, best_path = run / "last.pt", run / "best.pt"
        if sha(last_path) != report["last_checkpoint_sha256"] or sha(best_path) != report["best_checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {seed}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        state_finite = all(bool(torch.all(torch.isfinite(tensor))) for tensor in checkpoint["model_state"].values())
        if not state_finite:
            raise FloatingPointError(f"Nonfinite last.pt weights for {seed}")
        last_aggregate = ledger[-1]["aggregate"]
        members.append({"seed": seed, "exit_code": entry["exit_code"], "checks": checks,
            "history_rows": len(history), "validation_ledger_entries": len(ledger),
            "best_cycle": report["best_cycle"], "last_validation_objective_total": last_aggregate["objective_metrics"]["total"],
            "last_validation_full_sphere_lsd_db": last_aggregate["aggregate_full_sphere_lsd_db"],
            "last_checkpoint_sha256": report["last_checkpoint_sha256"], "last_checkpoint_state_all_finite": True,
            "elapsed_seconds": report["elapsed_seconds"], "peak_cuda_allocated_mib": report["peak_cuda_allocated_mib"]})
    payload = {"schema_version": "1.0", "status": "passed", "verified_on": "2026-09-05",
        "members": members, "member_count": 3, "all_exit_zero": True,
        "all_histories_ledgers_and_last_weights_finite": True, "cycles_per_member": 190,
        "optimizer_steps_per_member": 49780, "total_optimizer_steps": 149340,
        "checkpoint_policy": "cycle-190 last.pt only", "test_subjects_read": 0,
        "training_git_commit": "bd00151f7fc717fc748454e5419fc16d3f65d06e",
        "next_action": "Freeze the three-member last.pt ensemble and evaluate validation only."}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
