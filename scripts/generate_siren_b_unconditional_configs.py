"""Generate the three frozen Stage B unconditional baseline configs."""

from __future__ import annotations

import json

from mcar.paths import project_root


SEEDS = (20260821, 20260822, 20260823)
PROTOCOL = "experiments/film_siren/STAGE_B_UNCONDITIONAL_BASELINE_PROTOCOL.md"


def configuration(seed: int) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "created_on": "2026-08-22",
        "experiment_id": f"SIREN-B-UNCONDITIONAL-SEED-{seed}",
        "model_version": f"shared-siren-b-unconditional-seed-{seed}-v1",
        "experiment_type": "joint_unconditional_siren_stage_b",
        "protocol": PROTOCOL,
        "search_stage": "unconditional_baseline",
        "dataset_root": "data/processed/sonicom_residual_q26_v1",
        "subject_split_csv": "configs/data/sonicom_subject_split_v1.csv",
        "direction_protocol": "train_and_validate_on_767_interpolation_only_exclude_q26",
        "frequency_mapping": {
            "mode": "dual",
            "formula_version": "dual_linear_erb_v1",
            "frequency_minimum_hz": 86.1328125,
            "frequency_maximum_hz": 19982.8125,
            "coordinate_minimum": -1.0,
            "coordinate_maximum": 1.0,
        },
        "model": {
            "input_dimension": 5,
            "hidden_width": 256,
            "sine_layer_count": 6,
            "first_omega": 20.0,
            "hidden_omega": 20.0,
            "output_dimension": 2,
        },
        "optimizer": {"name": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0},
        "objective": "normalized_residual_mse",
        "cycles": 100,
        "steps_per_cycle": 262,
        "directions_per_step": 16,
        "validation_interval_cycles": 5,
        "validation_directions_per_block": 32,
        "gradient_clip": 5.0,
        "automatic_mixed_precision": False,
        "precision": "FP32",
        "seed": seed,
        "require_cuda": True,
        "require_clean_git": True,
        "condition_policy": "No Q26 magnitude, indices, normalization, encoder, latent, or FiLM parameters are read or constructed.",
        "test_policy": "The entry point resolves only the 262 train and 44 validation rows from the frozen split CSV. Test paths are never constructed or opened.",
        "run_name": f"sonicom_shared_siren_b_unconditional_seed{seed}",
    }


def main() -> None:
    root = project_root()
    outputs: list[str] = []
    for seed in SEEDS:
        output = (
            root
            / "configs"
            / "experiments"
            / f"sonicom_shared_siren_b_unconditional_seed{seed}.json"
        )
        output.write_text(json.dumps(configuration(seed), indent=2) + "\n", encoding="utf-8")
        outputs.append(str(output))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
