"""Generate frozen validation manifests for the Q14/Q50 FSC ensembles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260821, 20260822, 20260823)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def identity(payload: dict) -> str:
    values = dict(payload)
    values.pop("identity_sha256", None)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


for q in (14, 50):
    members = []
    for index, seed in enumerate(SEEDS, start=1):
        film_cfg = json.loads((ROOT / "configs/experiments" / f"sonicom_fsc_q{q}_film_seed{seed}_e130.json").read_text())
        film_run = film_cfg["run_name"]
        cnn_cfg_path = ROOT / "configs/experiments" / f"sonicom_fsc_q{q}_cnn_seed{seed}_e190.json"
        cnn_cfg = json.loads(cnn_cfg_path.read_text())
        run = cnn_cfg["run_name"]
        report = json.loads((ROOT / "artifacts/training" / run / "training_report.json").read_text())
        checkpoint = ROOT / "artifacts/training" / run / "last.pt"
        if report.get("status") != "completed" or report.get("cycles") != 190 or report.get("test_subjects_read") != 0:
            raise RuntimeError(f"invalid CNN report for {run}")
        if sha256(checkpoint) != report["authoritative_checkpoint_sha256"]:
            raise RuntimeError(f"CNN checkpoint hash mismatch for {run}")
        members.append({
            "member_index": index,
            "seed": seed,
            "weight": 1.0 / 3.0,
            "config": f"configs/experiments/sonicom_fsc_q{q}_cnn_seed{seed}_e190.json",
            "config_sha256": sha256(cnn_cfg_path),
            "checkpoint": f"artifacts/training/{run}/last.pt",
            "checkpoint_sha256": sha256(checkpoint),
            "training_report": f"artifacts/training/{run}/training_report.json",
            "training_git_commit": report["git"]["commit"],
            "training_git_dirty": report["git"]["dirty"],
        })
    payload = {
        "schema_version": "1.0", "status": "frozen", "created_on": "2026-09-07",
        "model_version": f"fsc-q{q}-e190-ensemble-v1",
        "model_family": "FiLM-SIREN spectral-CNN residual ensemble",
        "e_final": 190,
        "ensemble": {"type": "residual_db_mean", "member_count": 3, "weights": [1/3]*3, "checkpoint_policy": "cycle-190 last.pt only"},
        "members": members,
        "dataset": {
            "root": f"data/processed/sonicom_fsc_q{q}_residual_v1",
            "split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "q26_csv": "configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.csv",
            "q26_normalization": f"configs/data/sonicom_fsc_q{q}_magnitude_normalization_v1.json",
            "observation_count": q, "split": "val", "subject_count": 44,
            "tensor_layout": "ear,direction,frequency",
        },
        "test_policy": "Validation only; test paths are never constructed or read.",
    }
    payload["identity_sha256"] = identity(payload)
    out = ROOT / "configs/experiments" / f"sonicom_fsc_q{q}_e190_ensemble_manifest.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out.relative_to(ROOT), payload["identity_sha256"])
