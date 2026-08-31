"""Generate the authorized hash-locked Hybrid E190 test manifest and registry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mcar.frozen_test import file_sha256, identity_sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_test_manifest.json"
REGISTRY = ROOT / "configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_test_registry.json"
SOURCE = ROOT / "configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json"
SEEDS = (20260821, 20260822, 20260823)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resource(path: str, purpose: str) -> dict[str, str]:
    target = ROOT / path
    if not target.is_file():
        raise FileNotFoundError(target)
    return {"path": path, "sha256": file_sha256(target), "purpose": purpose}


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def main() -> None:
    dirty = [
        line[3:]
        for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
        if line[3:] != ".gitignore"
    ]
    if dirty:
        raise RuntimeError(f"Manifest generation requires relevant tracked files clean: {dirty}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source["identity_sha256"] != "A3CFAC9C206E0A53FD2FA130817673AAFE07B66855322B5D34824E9173BDCCFE":
        raise ValueError("Frozen validation identity changed")
    members = []
    for member, seed in zip(source["members"], SEEDS):
        if member["seed"] != seed:
            raise ValueError("Unexpected member seed")
        if file_sha256(ROOT / member["checkpoint"]) != member["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        members.append(dict(member))

    dataset_resources = [
        resource("configs/data/sonicom_subject_split_v1.csv", "frozen 262/44/44 split"),
        resource("configs/data/sonicom_sparse_grid_q26_v1.csv", "Q26 sparse grid"),
        resource("configs/data/siren_b_q26_normalization_v1.json", "train-only Q26 normalization"),
        resource("data/processed/sonicom_residual_q26_v1/training_statistics.json", "train-only residual normalization"),
    ]
    evaluator_resources = [
        resource("src/mcar/evaluation/predict_film_siren_spectral_cnn_frozen_test.py", "single authorized predictor"),
        resource("src/mcar/predictors.py", "Hybrid E190 predictor"),
        resource("src/mcar/frozen_test.py", "manifest and registry guards"),
        resource("matlab/+mcar/evaluate_sonicom_interpolation_baselines.m", "strict test evaluator"),
        resource("scripts/analyze_film_siren_spectral_cnn_frozen_test.py", "frozen ten-method merge and bootstrap"),
        resource("experiments/film_siren/STAGE_D_HYBRID_E190_FROZEN_TEST_PROTOCOL.md", "test preregistration"),
        resource("results/sonicom_bounded_mcar_film_correction_final_e25_frozen_test_nine_method/metric_long.csv", "frozen nine-method test metrics"),
    ]
    now = "2026-08-31T03:49:13Z"
    manifest = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_at_utc": now,
        "model_version": "film-siren-spectral-cnn-final-e190-ensemble-test-v1",
        "model_family": source["model_family"],
        "e_final": 190,
        "scheduler_horizon_cycles": 200,
        "ensemble": source["ensemble"],
        "members": members,
        "dataset": {
            "root": source["dataset"]["root"],
            "split_csv": source["dataset"]["split_csv"],
            "q26_csv": source["dataset"]["q26_csv"],
            "q26_normalization": source["dataset"]["q26_normalization"],
            "split": "test",
            "subject_count": 44,
            "resources": dataset_resources,
        },
        "evaluator": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty_at_generation": False,
            "preexisting_ignored_dirty_paths": [".gitignore"],
            "resources": evaluator_resources,
        },
        "comparison": {
            "final_method_count": 10,
            "paired_baselines": ["BOUNDED", "MCARv351", "RANF", "FSPAE", "MCA"],
            "metrics": list(METRICS),
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260831,
        },
        "authorization": {
            "user_text": "目前FiLM-SIREN + CNN（Hybrid E190）是不是没有在test集上跑过，如果没有的话可以跑一次",
            "recorded_at_local": "2026-08-31T11:49:13+08:00",
        },
        "recovery_policy": "Only proven incomplete failure under unchanged identity; no tuning or retest",
    }
    manifest["identity_sha256"] = identity_sha256(manifest)
    registry = {
        "schema_version": "1.0",
        "created_at_utc": now,
        "updated_at_utc": now,
        "task_id": "film-siren-spectral-cnn-final-e190-frozen-test-v1",
        "state": "not_started",
        "manifest": rel(MANIFEST),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "authorization": manifest["authorization"],
        "retry_count": 0,
        "test_subject_count_read": 0,
        "recovery_policy": manifest["recovery_policy"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(manifest["identity_sha256"])


METRICS = (
    "FullSphereERB",
    "Contralateral25ERB",
    "ContralateralHighFrequency",
    "HorizontalILDMAE",
)


if __name__ == "__main__":
    main()

