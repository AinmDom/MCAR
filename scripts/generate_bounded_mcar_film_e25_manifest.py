"""Freeze the formal Stage-E E25 bounded-correction ensemble manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260821, 20260822, 20260823)
E_FINAL = 25
RUN_PREFIX = "sonicom_bounded_mcar_film_correction_final_seed"
OUTPUT = ROOT / (
    "configs/experiments/"
    "sonicom_bounded_mcar_film_correction_final_e25_ensemble_manifest.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def identity(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resource(path: str, purpose: str) -> dict[str, str]:
    absolute = ROOT / path
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return {"path": path, "sha256": sha256(absolute), "purpose": purpose}


def git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def validate_member(seed: int, index: int) -> dict[str, object]:
    run_name = f"{RUN_PREFIX}{seed}_e{E_FINAL}"
    run_root = ROOT / "artifacts/training" / run_name
    config = ROOT / "configs/experiments" / f"{run_name}.json"
    checkpoint = run_root / "last.pt"
    report_path = run_root / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "status": "completed",
        "run_name": run_name,
        "seed": seed,
        "cycles": E_FINAL,
        "optimizer_steps": 6550,
        "decision": "FIXED_CYCLE_COMPLETE",
        "formal_fixed_cycle": True,
        "checkpoint_policy": "fixed_stop_cycle_last",
        "authoritative_checkpoint": "last.pt",
        "scheduler_horizon_cycles": 40,
        "test_subjects_read": 0,
        "conditioning_scope": "global_plus_local_mca",
        "model_family": "BoundedMcarFilmCorrection",
        "film_siren_frozen": True,
        "trainable_parameter_count": 514,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"{report_path}: {key} != {expected!r}")
    if report["git"]["dirty"] is not False:
        raise ValueError(f"Training worktree was dirty for seed {seed}")

    with (run_root / "history.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        history = list(csv.DictReader(handle))
    if len(history) != E_FINAL:
        raise ValueError(f"Expected {E_FINAL} history rows for seed {seed}")
    numeric_history = [
        float(value)
        for row in history
        for key, value in row.items()
        if key != "cycle" and value not in (None, "")
    ]
    if not all(math.isfinite(value) for value in numeric_history):
        raise FloatingPointError(f"Non-finite history value for seed {seed}")

    ledger = json.loads(
        (run_root / "validation_ledger.json").read_text(encoding="utf-8")
    )
    if [int(row["cycle"]) for row in ledger] != [5, 10, 15, 20, 25]:
        raise ValueError(f"Unexpected validation ledger for seed {seed}")

    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != report["last_checkpoint_sha256"]:
        raise ValueError(f"Last checkpoint hash mismatch for seed {seed}")
    if checkpoint_hash != report["authoritative_checkpoint_sha256"]:
        raise ValueError(f"Authoritative checkpoint hash mismatch for seed {seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors = [value for value in payload["model_state"].values() if torch.is_tensor(value)]
    if len(tensors) != 202 or not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError(f"Invalid checkpoint tensors for seed {seed}")

    stderr = ROOT / "artifacts/training_logs" / f"{run_name}.stderr.log"
    if not stderr.is_file() or stderr.stat().st_size != 0:
        raise ValueError(f"Non-empty or missing stderr for seed {seed}")
    return {
        "member_index": index,
        "seed": seed,
        "weight": 1.0 / len(SEEDS),
        "config": relative(config),
        "config_sha256": sha256(config),
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "training_report": relative(report_path),
        "training_git_commit": report["git"]["commit"],
        "training_git_dirty": report["git"]["dirty"],
        "checkpoint_cycle": int(payload["cycle"]),
        "checkpoint_tensor_count": len(tensors),
        "checkpoint_all_finite": True,
        "history_rows": len(history),
        "ledger_cycles": [5, 10, 15, 20, 25],
        "stderr_bytes": 0,
        "test_subjects_read": 0,
    }


def main() -> None:
    members = [validate_member(seed, index) for index, seed in enumerate(SEEDS, 1)]
    manifest = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_on": "2026-08-29",
        "model_version": "bounded-mcar-film-correction-final-e25-ensemble-v1",
        "model_family": "Bounded MCAR plus FiLM-SIREN correction ensemble",
        "e_final": E_FINAL,
        "scheduler_horizon_cycles": 40,
        "ensemble": {
            "type": "residual_db_mean",
            "member_count": 3,
            "weights": [1.0 / 3.0] * 3,
            "checkpoint_policy": "cycle-25 last.pt only",
        },
        "members": members,
        "dataset": {
            "root": "data/processed/sonicom_residual_q26_v1",
            "split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "q26_csv": "configs/data/sonicom_sparse_grid_q26_v1.csv",
            "q26_normalization": "configs/data/siren_b_q26_normalization_v1.json",
            "split": "val",
            "subject_count": 44,
            "tensor_layout": "ear,direction,frequency",
        },
        "evaluator": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty_at_generation": bool(git_value("status", "--porcelain")),
            "resources": [
                resource(
                    "src/mcar/evaluation/predict_bounded_mcar_film_correction_ensemble.py",
                    "hash-guarded three-member validation predictor",
                ),
                resource("src/mcar/predictors.py", "bounded-correction member predictor"),
                resource(
                    "matlab/+mcar/evaluate_film_siren_spectral_cnn_d1_validation.m",
                    "strict residual-method evaluator",
                ),
                resource(
                    "experiments/film_siren/STAGE_E_BOUNDED_CORRECTION_FORMAL_E25_FREEZE.md",
                    "formal budget and equal-weight freeze",
                ),
            ],
        },
        "comparison": {
            "methods": ["BOUNDEDENS", "HYBRID", "FILMENS", "MCAR"],
            "metrics": [
                "FullSphereERB",
                "Contralateral25ERB",
                "ContralateralHighFrequency",
                "HorizontalILDMAE",
            ],
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260828,
            "difference": "bounded ensemble minus baseline; negative favors bounded ensemble",
        },
        "test_policy": "Validation only; test paths are never constructed or read.",
    }
    manifest["identity_sha256"] = identity(manifest)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(relative(OUTPUT))
    print(manifest["identity_sha256"])


if __name__ == "__main__":
    main()
