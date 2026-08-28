"""Freeze the formal E190 spectral-CNN ensemble manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "configs"
    / "experiments"
    / "sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json"
)
SEEDS = (20260821, 20260822, 20260823)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def identity(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def main() -> None:
    members = []
    for index, seed in enumerate(SEEDS, start=1):
        config = Path(
            f"configs/experiments/"
            f"sonicom_film_siren_spectral_cnn_final_seed{seed}_e190.json"
        )
        run_root = Path(
            f"artifacts/training/"
            f"sonicom_film_siren_spectral_cnn_final_seed{seed}_e190"
        )
        checkpoint = run_root / "last.pt"
        report_path = run_root / "training_report.json"
        report = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
        if (
            report["status"] != "completed"
            or report["decision"] != "FIXED_CYCLE_COMPLETE"
            or report["cycles"] != 190
            or report["authoritative_checkpoint"] != "last.pt"
            or report["test_subjects_read"] != 0
        ):
            raise ValueError(f"Invalid formal report for seed {seed}")
        checkpoint_hash = sha256(ROOT / checkpoint)
        if checkpoint_hash != report["last_checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
        members.append(
            {
                "member_index": index,
                "seed": seed,
                "weight": 1.0 / 3.0,
                "config": str(config).replace("\\", "/"),
                "config_sha256": sha256(ROOT / config),
                "checkpoint": str(checkpoint).replace("\\", "/"),
                "checkpoint_sha256": checkpoint_hash,
                "training_report": str(report_path).replace("\\", "/"),
                "training_git_commit": report["git"]["commit"],
                "training_git_dirty": report["git"]["dirty"],
            }
        )

    payload = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_on": "2026-08-28",
        "model_version": "film-siren-spectral-cnn-final-e190-ensemble-v1",
        "model_family": "FiLM-SIREN spectral-CNN residual ensemble",
        "e_final": 190,
        "scheduler_horizon_cycles": 200,
        "ensemble": {
            "type": "residual_db_mean",
            "member_count": 3,
            "weights": [1.0 / 3.0] * 3,
            "checkpoint_policy": "cycle-190 last.pt only",
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
        "comparison": {
            "methods": ["HYBRID", "PARENT", "FILMENS", "MCAR"],
            "metrics": [
                "FullSphereERB",
                "Contralateral25ERB",
                "ContralateralHighFrequency",
                "HorizontalILDMAE",
            ],
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260828,
            "difference": "HYBRID minus baseline; negative favors HYBRID",
        },
        "test_policy": "Validation only; test paths are never constructed or read.",
    }
    payload["identity_sha256"] = identity(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    print(payload["identity_sha256"])


if __name__ == "__main__":
    main()
