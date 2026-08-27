"""Generate the hash-locked final FiLM-SIREN ensemble manifest and registry."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mcar.frozen_test import file_sha256, identity_sha256


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/sonicom_film_siren_gl_final_cosine_c0_e140_ensemble_manifest.json"
REGISTRY_PATH = ROOT / "configs/experiments/sonicom_film_siren_gl_final_test_registry.json"
SEEDS = (20260821, 20260822, 20260823)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resource(path: str, purpose: str) -> dict[str, str]:
    absolute = ROOT / path
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return {"path": path, "sha256": file_sha256(absolute), "purpose": purpose}


def git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    evaluator_commit = git_value("rev-parse", "HEAD")
    members = []
    for index, seed in enumerate(SEEDS, start=1):
        run_name = f"sonicom_film_siren_gl_final_cosine_c0_seed{seed}_e140"
        config = ROOT / f"configs/experiments/{run_name}.json"
        run_root = ROOT / "artifacts/training" / run_name
        checkpoint = run_root / "last.pt"
        report_path = run_root / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        required = {
            "status": "completed",
            "seed": seed,
            "cycles": 140,
            "decision": "FIXED_CYCLE_COMPLETE",
            "formal_fixed_cycle": True,
            "checkpoint_policy": "fixed_stop_cycle_last",
            "authoritative_checkpoint": "last.pt",
            "scheduler_horizon_cycles": 150,
            "test_subjects_read": 0,
            "conditioning_scope": "global_plus_local_mca",
        }
        for key, expected in required.items():
            if report.get(key) != expected:
                raise ValueError(f"{report_path}: {key} != {expected!r}")
        checkpoint_hash = file_sha256(checkpoint)
        if checkpoint_hash != report["authoritative_checkpoint_sha256"]:
            raise ValueError(f"Authoritative checkpoint hash mismatch: {checkpoint}")
        members.append(
            {
                "member_index": index,
                "seed": seed,
                "weight": 1.0 / len(SEEDS),
                "config": relative(config),
                "config_sha256": file_sha256(config),
                "checkpoint": relative(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "training_report": relative(report_path),
                "training_git_commit": report["git"]["commit"],
                "training_git_dirty": report["git"]["dirty"],
            }
        )

    dataset_resources = [
        resource("configs/data/sonicom_subject_split_v1.csv", "262/44/44 frozen split"),
        resource("configs/data/sonicom_sparse_grid_q26_v1.csv", "Q26 sparse grid"),
        resource("configs/data/siren_b_q26_normalization_v1.json", "train-only Q26 normalization"),
        resource("data/processed/sonicom_residual_q26_v1/training_statistics.json", "train-only residual normalization"),
    ]
    evaluator_resources = [
        resource("src/mcar/evaluation/predict_film_siren_ensemble.py", "hash-guarded ensemble predictor"),
        resource("src/mcar/predictors.py", "seven-dimensional member predictor"),
        resource("src/mcar/frozen_test.py", "manifest and atomic registry guards"),
        resource("matlab/+mcar/evaluate_sonicom_validation_reconstruction.m", "strict reconstruction evaluator"),
        resource("scripts/analyze_siren_gl_frozen_test.py", "pre-registered paired bootstrap gates"),
        resource("configs/experiments/sonicom_mlp_cnn_q26_v351.json", "frozen MCAR v3.5.1 baseline definition"),
        resource("experiments/film_siren/EXPERIMENT_CHECKLIST.md", "pre-registered engineering test policy"),
        resource("experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_FINAL_TRAINING_FREEZE.md", "final training freeze"),
        resource("results/sonicom_film_siren_gl_top3/final_selection.json", "three-seed final selection"),
    ]
    baseline_path = ROOT / "configs/experiments/sonicom_mlp_cnn_q26_v351.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_at_utc": now,
        "model_version": "film-siren-gl-cosine-c0-e140-ensemble-v1",
        "model_family": "FiLM-SIREN residual ensemble",
        "selection": "cosine+C0 by three-seed validation mean",
        "e_final": 140,
        "scheduler_horizon_cycles": 150,
        "ensemble": {
            "type": "residual_db_mean",
            "member_count": 3,
            "weights": [1.0 / 3.0] * 3,
            "checkpoint_policy": "cycle-140 last.pt only",
        },
        "members": members,
        "dataset": {
            "root": "data/processed/sonicom_residual_q26_v1",
            "split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "q26_csv": "configs/data/sonicom_sparse_grid_q26_v1.csv",
            "q26_normalization": "configs/data/siren_b_q26_normalization_v1.json",
            "schema": {
                "subject_counts": {"train": 262, "val": 44, "test": 44},
                "subject_file": "subjects/<subject_id>/q26.h5",
                "tensor_layout": "ear,direction,frequency",
                "ear_count": 2,
                "direction_count": 793,
                "frequency_count": 463,
                "q26_direction_count": 26,
                "interpolation_direction_count": 767,
                "independent_horizontal_interpolation_count": 72,
            },
            "frequency_mapping": {
                "mode": "dual",
                "formula_version": "dual_linear_erb_v1",
                "minimum_hz": 86.1328125,
                "maximum_hz": 19982.8125,
                "coordinate_range": [-1.0, 1.0],
            },
            "interpolation_mask_policy": "strict interpolation_only; sparse Q26 directions excluded",
            "resources": dataset_resources,
        },
        "evaluator": {
            "git_commit": evaluator_commit,
            "git_dirty_at_generation": bool(git_value("status", "--porcelain")),
            "resources": evaluator_resources,
        },
        "comparison": {
            "baseline_model_version": baseline["model_version"],
            "baseline_manifest": relative(baseline_path),
            "baseline_manifest_sha256": file_sha256(baseline_path),
            "baseline_formula": baseline["formula"],
            "scope": "historically consumed frozen SONICOM engineering test; not independent confirmation",
        },
        "statistics": {
            "paired_difference": "FiLM-SIREN minus MCAR v3.5.1; negative is better",
            "primary_endpoint": "interpolation-only solid-angle-weighted Full-sphere ERB",
            "bootstrap": {
                "subject_rows": 44,
                "replicates": 10000,
                "seed": 20260818,
                "interval": "two-sided percentile 95% CI",
            },
            "primary_gate": "upper CI < 0 dB",
            "secondary_non_inferiority_margins_db": {
                "contralateral_25_erb": 0.02,
                "contralateral_high_frequency": 0.05,
                "strict_horizontal_ild": 0.02,
            },
            "secondary_gate": "every upper CI < its pre-registered margin",
            "distribution_gate": "Full ERB candidate wins >=26/44; engineering majority only, not sign-test significance",
        },
        "planned_commands": {
            "prediction": (
                "python -m mcar.evaluation.predict_film_siren_ensemble "
                "configs/experiments/sonicom_film_siren_gl_final_cosine_c0_e140_ensemble_manifest.json "
                "sonicom_film_siren_gl_final_cosine_c0_e140_ensemble_test --split test --allow-test "
                "--registry configs/experiments/sonicom_film_siren_gl_final_test_registry.json"
            ),
            "strict_reconstruction": (
                "mcar.evaluate_sonicom_validation_reconstruction(inf, "
                "'sonicom_film_siren_gl_final_vs_v351_frozen_test', true, "
                "'sonicom_q26_test_v351_previous30_b70', "
                "'sonicom_film_siren_gl_final_cosine_c0_e140_ensemble_test', "
                "'FiLM-SIREN GL cosine C0 E140 1/3 ensemble', 'test', "
                "'sonicom_q26_test_v1', 'sonicom_q26_test_v2', true)"
            ),
            "statistical_analysis": (
                "python scripts/analyze_siren_gl_frozen_test.py <per_subject_metrics.csv> "
                "results/sonicom_film_siren_gl_final_vs_v351_frozen_test/statistical_decision.json"
            ),
        },
    }
    manifest["identity_sha256"] = identity_sha256(manifest)
    registry = {
        "schema_version": "1.0",
        "created_at_utc": now,
        "updated_at_utc": now,
        "task_id": "sonicom-film-siren-gl-final-vs-v351-frozen-engineering-test-v1",
        "state": "not_started",
        "manifest": relative(MANIFEST_PATH),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "model_identity": {
            "config_sha256": [member["config_sha256"] for member in members],
            "checkpoint_sha256": [member["checkpoint_sha256"] for member in members],
        },
        "authorization": None,
        "retry_count": 0,
        "test_subject_count_read": 0,
        "recovery_policy": "Only a proven incomplete failure with unchanged manifest identity may be resumed",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {relative(MANIFEST_PATH)}")
    print(f"wrote {relative(REGISTRY_PATH)}")
    print(f"identity {manifest['identity_sha256']}")


if __name__ == "__main__":
    main()
