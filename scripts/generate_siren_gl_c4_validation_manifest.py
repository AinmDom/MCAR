"""Generate the hash-locked E130 D1/D2+notch FiLM-SIREN ensemble manifest.

Validation-only: no test registry is created because this round has no test
authorization. The manifest is used to run the three-member 1/3 residual-dB
ensemble prediction over the 44 locked validation subjects.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mcar.frozen_test import file_sha256, identity_sha256


ROOT = Path(__file__).resolve().parents[1]
SELECTION = (
    "results/sonicom_film_siren_gl_c4_corrected_expansion/selection.json"
)
MANIFEST_PATH = ROOT / (
    "configs/experiments/"
    "sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_manifest.json"
)
SEEDS = (20260821, 20260822, 20260823)
RUN_PREFIX = "sonicom_film_siren_gl_final_d1d2_notch_seed"
E_FINAL = 130


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
    selection = json.loads((ROOT / SELECTION).read_text(encoding="utf-8"))
    if selection["candidate"] != "warmup_cosine_d1d2_notch":
        raise ValueError("selection candidate mismatch")
    if int(selection["e_final"]) != E_FINAL:
        raise ValueError("selection e_final mismatch")
    if [int(s) for s in selection["formal_seeds"]] != list(SEEDS):
        raise ValueError("selection formal_seeds mismatch")

    evaluator_commit = git_value("rev-parse", "HEAD")
    members = []
    for index, seed in enumerate(SEEDS, start=1):
        run_name = f"{RUN_PREFIX}{seed}_e{E_FINAL}"
        config = ROOT / f"configs/experiments/{run_name}.json"
        run_root = ROOT / "artifacts/training" / run_name
        checkpoint = run_root / "last.pt"
        report_path = run_root / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        required = {
            "status": "completed",
            "seed": seed,
            "cycles": E_FINAL,
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
        resource(
            "configs/data/sonicom_subject_split_v1.csv",
            "262/44/44 frozen split",
        ),
        resource(
            "configs/data/sonicom_sparse_grid_q26_v1.csv",
            "Q26 sparse grid",
        ),
        resource(
            "configs/data/siren_b_q26_normalization_v1.json",
            "train-only Q26 normalization",
        ),
        resource(
            "data/processed/sonicom_residual_q26_v1/training_statistics.json",
            "train-only residual normalization",
        ),
    ]
    evaluator_resources = [
        resource(
            "src/mcar/evaluation/predict_film_siren_ensemble.py",
            "hash-guarded ensemble predictor",
        ),
        resource(
            "src/mcar/predictors.py",
            "seven-dimensional member predictor",
        ),
        resource(
            "src/mcar/frozen_test.py",
            "manifest and atomic registry guards",
        ),
        resource(
            "matlab/+mcar/evaluate_film_siren_d1d2_notch_four_method_validation.m",
            "strict four-method reconstruction evaluator",
        ),
        resource(
            "scripts/analyze_siren_gl_c4_four_method_validation.py",
            "pre-registered four-method paired bootstrap summary",
        ),
        resource(
            "configs/experiments/sonicom_mlp_cnn_q26_v351.json",
            "frozen MCAR v3.5.1 baseline definition",
        ),
        resource(
            "experiments/film_siren/"
            "STAGE_C_GLOBAL_LOCAL_MCA_C4_CORRECTED_FORMAL_TRAINING_FREEZE.md",
            "corrected formal training freeze",
        ),
        resource(
            "experiments/film_siren/"
            "STAGE_C_GLOBAL_LOCAL_MCA_C4_COMMON_SCORE_CORRECTION.md",
            "common-score correction amendment",
        ),
        resource(
            "results/sonicom_film_siren_gl_c4_corrected_expansion/selection.json",
            "three-seed corrected selection",
        ),
    ]
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_at_utc": now,
        "model_version": "film-siren-gl-d1d2-notch-e130-ensemble-v1",
        "model_family": "FiLM-SIREN residual ensemble",
        "selection": "warmup_cosine_d1d2_notch by corrected three-seed expansion",
        "e_final": E_FINAL,
        "scheduler_horizon_cycles": 150,
        "ensemble": {
            "type": "residual_db_mean",
            "member_count": 3,
            "weights": [1.0 / 3.0] * 3,
            "checkpoint_policy": f"cycle-{E_FINAL} last.pt only",
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
            "interpolation_mask_policy": (
                "strict interpolation_only; sparse Q26 directions excluded"
            ),
            "resources": dataset_resources,
        },
        "evaluator": {
            "git_commit": evaluator_commit,
            "git_dirty_at_generation": bool(git_value("status", "--porcelain")),
            "resources": evaluator_resources,
        },
        "comparison": {
            "baseline_model_version": "v3.5.1",
            "baseline_manifest": "configs/experiments/sonicom_mlp_cnn_q26_v351.json",
            "baseline_manifest_sha256": file_sha256(
                ROOT / "configs/experiments/sonicom_mlp_cnn_q26_v351.json"
            ),
            "baseline_formula": (
                "predicted_residual_v351 = "
                "0.3 * previous_joint + 0.7 * v351b_continuation"
            ),
            "scope": (
                "validation-only horizontal comparison with MCAR v3.5.1, "
                "RANF and FSP-AE; no test subjects are read"
            ),
        },
        "statistics": {
            "paired_difference": (
                "method minus baseline; negative is better"
            ),
            "bootstrap": {
                "subject_rows": 44,
                "replicates": 10000,
                "seed": 20260819,
                "interval": "two-sided percentile 95% CI",
            },
        },
        "planned_commands": {
            "prediction": (
                "python -m mcar.evaluation.predict_film_siren_ensemble "
                "configs/experiments/"
                "sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_manifest.json "
                "sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_validation "
                "--split val"
            ),
            "strict_reconstruction": (
                "mcar.evaluate_film_siren_d1d2_notch_four_method_validation("
                "inf, 'sonicom_film_siren_gl_final_d1d2_notch_"
                "vs_ranf_fsp_v351_validation', "
                "'sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_validation', "
                "'sonicom_q26_validation_v351_previous30_b70', "
                "'sonicom_ranf_q26_validation_frozen', "
                "'sonicom_fsp_ae_q26_formal_validation')"
            ),
            "statistical_analysis": (
                "python scripts/analyze_siren_gl_c4_four_method_validation.py "
                "<metric_long.csv> "
                "results/sonicom_film_siren_gl_final_d1d2_notch_"
                "vs_ranf_fsp_v351_validation/four_method_decision.json"
            ),
        },
    }
    manifest["identity_sha256"] = identity_sha256(manifest)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {relative(MANIFEST_PATH)}")
    print(f"identity {manifest['identity_sha256']}")


if __name__ == "__main__":
    main()
