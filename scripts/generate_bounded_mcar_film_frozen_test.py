"""Generate the authorized hash-locked Stage-E test manifest and registry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mcar.frozen_test import file_sha256, identity_sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiments/sonicom_bounded_mcar_film_correction_final_e25_test_manifest.json"
REGISTRY = ROOT / "configs/experiments/sonicom_bounded_mcar_film_correction_final_e25_test_registry.json"
SOURCE_MANIFEST = ROOT / "configs/experiments/sonicom_bounded_mcar_film_correction_final_e25_ensemble_manifest.json"
SEEDS = (20260821, 20260822, 20260823)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resource(path: str, purpose: str) -> dict[str, str]:
    target = ROOT / path
    if not target.is_file():
        raise FileNotFoundError(target)
    return {"path": path, "sha256": file_sha256(target), "purpose": purpose}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    if git("status", "--porcelain"):
        raise RuntimeError("Manifest generation requires a clean tracked worktree")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source["identity_sha256"] != "72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705":
        raise ValueError("Frozen validation identity changed")
    members = []
    for member, seed in zip(source["members"], SEEDS, strict=True):
        if member["seed"] != seed or member["checkpoint_cycle"] != 25:
            raise ValueError("Unexpected member")
        if file_sha256(ROOT / member["checkpoint"]) != member["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        members.append({k: member[k] for k in ("member_index", "seed", "weight", "config",
                                               "config_sha256", "checkpoint", "checkpoint_sha256",
                                               "training_report", "training_git_commit", "training_git_dirty")})
    dataset_resources = [
        resource("configs/data/sonicom_subject_split_v1.csv", "frozen 262/44/44 split"),
        resource("configs/data/sonicom_sparse_grid_q26_v1.csv", "Q26 sparse grid"),
        resource("configs/data/siren_b_q26_normalization_v1.json", "train-only Q26 normalization"),
        resource("data/processed/sonicom_residual_q26_v1/training_statistics.json", "train-only residual normalization"),
    ]
    evaluator_resources = [
        resource("src/mcar/evaluation/predict_bounded_mcar_film_correction_frozen_test.py", "single authorized predictor"),
        resource("src/mcar/predictors.py", "bounded correction predictor"),
        resource("src/mcar/frozen_test.py", "manifest and registry guards"),
        resource("matlab/+mcar/evaluate_sonicom_interpolation_baselines.m", "strict test evaluator"),
        resource("scripts/analyze_bounded_mcar_film_frozen_test.py", "frozen nine-method merge and bootstrap"),
        resource("experiments/film_siren/STAGE_E_BOUNDED_CORRECTION_FROZEN_TEST_PROTOCOL.md", "test preregistration"),
        resource("results/sonicom_eight_method_q26_v351_main_test/metric_long.csv", "frozen eight-method test metrics"),
    ]
    now = "2026-08-29T15:31:38.3013460Z"
    manifest = {
        "schema_version": "1.0", "status": "frozen", "created_at_utc": now,
        "model_version": "bounded-mcar-film-correction-final-e25-ensemble-test-v1",
        "model_family": source["model_family"], "e_final": 25,
        "scheduler_horizon_cycles": 40, "ensemble": source["ensemble"], "members": members,
        "dataset": {"root": source["dataset"]["root"], "split_csv": source["dataset"]["split_csv"],
                    "q26_csv": source["dataset"]["q26_csv"],
                    "q26_normalization": source["dataset"]["q26_normalization"],
                    "split": "test", "subject_count": 44, "resources": dataset_resources},
        "evaluator": {"git_commit": git("rev-parse", "HEAD"), "git_dirty_at_generation": False,
                      "resources": evaluator_resources},
        "comparison": {"final_methods": ["BOUNDED", "MCARv351", "FSPAE", "RANF", "MCA",
                                                  "SUpDEqBary", "SUpDEqNN", "SUpDEqSH", "SHOnly"],
                       "paired_baselines": ["MCARv351", "RANF", "FSPAE", "MCA"],
                       "metrics": ["FullSphereERB", "Contralateral25ERB",
                                   "ContralateralHighFrequency", "HorizontalILDMAE"],
                       "bootstrap_replicates": 10000, "bootstrap_seed": 20260829},
        "authorization": {"user_text": "可以先做test评价", "recorded_at_local": "2026-08-29T23:31:38+08:00"},
        "recovery_policy": "Only proven incomplete failure under unchanged identity; no tuning or retest",
    }
    manifest["identity_sha256"] = identity_sha256(manifest)
    registry = {"schema_version": "1.0", "created_at_utc": now, "updated_at_utc": now,
                "task_id": "bounded-mcar-film-final-e25-frozen-test-v1", "state": "not_started",
                "manifest": rel(MANIFEST), "manifest_identity_sha256": manifest["identity_sha256"],
                "authorization": manifest["authorization"], "retry_count": 0,
                "test_subject_count_read": 0, "recovery_policy": manifest["recovery_policy"]}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(manifest["identity_sha256"])


if __name__ == "__main__":
    main()
