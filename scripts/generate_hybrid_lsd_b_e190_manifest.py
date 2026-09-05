"""Freeze the verified three-member LSD-B validation ensemble."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/experiments/sonicom_hybrid_lsd_b_e190_ensemble_manifest.json"
SEEDS = (20260821, 20260822, 20260823)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def identity(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    verification_path = ROOT / "reports/HYBRID_LSD_B_TRAINING_VERIFICATION_20260905.json"
    verification = json.loads(verification_path.read_text())
    if verification["status"] != "passed" or verification["test_subjects_read"] != 0:
        raise ValueError("Training verification has not passed")
    members = []
    for index, seed in enumerate(SEEDS, 1):
        config = Path(f"configs/experiments/sonicom_film_siren_spectral_cnn_lsd_b_seed{seed}_e190.json")
        run = Path(f"artifacts/training/sonicom_film_siren_spectral_cnn_lsd_b_seed{seed}_e190")
        report_path = run / "training_report.json"
        report = json.loads((ROOT / report_path).read_text())
        checkpoint = run / "last.pt"
        if sha(ROOT / checkpoint) != report["last_checkpoint_sha256"]:
            raise ValueError(seed)
        members.append({"member_index": index, "seed": seed, "weight": 1/3,
            "config": config.as_posix(), "config_sha256": sha(ROOT/config),
            "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": sha(ROOT/checkpoint),
            "training_report": report_path.as_posix(), "training_report_sha256": sha(ROOT/report_path),
            "training_git_commit": report["git"]["commit"], "training_git_dirty": report["git"]["dirty"]})
    payload = {"schema_version": "1.0", "status": "frozen", "created_on": "2026-09-05",
        "model_version": "film-siren-spectral-cnn-lsd-b-e190-ensemble-v1",
        "model_family": "FiLM-SIREN spectral-CNN residual ensemble with LSD loss", "e_final": 190,
        "scheduler_horizon_cycles": 200,
        "ensemble": {"type": "residual_db_mean", "member_count": 3, "weights": [1/3]*3, "checkpoint_policy": "cycle-190 last.pt only"},
        "members": members,
        "dataset": {"root": "data/processed/sonicom_residual_q26_v1", "split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "q26_csv": "configs/data/sonicom_sparse_grid_q26_v1.csv", "q26_normalization": "configs/data/siren_b_q26_normalization_v1.json",
            "split": "val", "subject_count": 44, "tensor_layout": "ear,direction,frequency"},
        "comparison": {"candidate": "HYBRID_LSD_B", "baseline": "HYBRID", "primary_endpoint": "FullSphereLSD",
            "bootstrap_replicates": 10000, "bootstrap_seed": 20260904, "tie_tolerance_db": 1e-12,
            "difference": "candidate minus baseline; negative favors candidate"},
        "training_verification": {"path": verification_path.relative_to(ROOT).as_posix(), "sha256": sha(verification_path)},
        "test_policy": "Validation only; test paths are never constructed or read."}
    payload["identity_sha256"] = identity(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT.relative_to(ROOT).as_posix(), "identity_sha256": payload["identity_sha256"]}))


if __name__ == "__main__":
    main()
