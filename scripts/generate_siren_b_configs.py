"""Generate the pre-registered Stage B latent-screening configurations."""

from __future__ import annotations

import json
from pathlib import Path

from mcar.paths import project_root


def configuration(latent_dimension: int) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "created_on": "2026-08-20",
        "experiment_id": f"SIREN-B-LATENT-{latent_dimension}",
        "model_version": f"film-siren-b-latent-{latent_dimension}-screen-v1",
        "experiment_type": "joint_film_siren_stage_b",
        "protocol": "experiments/film_siren/STAGE_B_FILM_PROTOCOL.md",
        "search_stage": "latent_dimension_screening",
        "dataset_root": "data/processed/sonicom_residual_q26_v1",
        "subject_split_csv": "configs/data/sonicom_subject_split_v1.csv",
        "q26_csv": "configs/data/sonicom_sparse_grid_q26_v1.csv",
        "q26_normalization": "configs/data/siren_b_q26_normalization_v1.json",
        "direction_protocol": "train_and_validate_on_767_interpolation_only_exclude_q26",
        "frequency_mapping": {
            "mode": "dual",
            "formula_version": "dual_linear_erb_v1",
            "frequency_minimum_hz": 86.1328125,
            "frequency_maximum_hz": 19982.8125,
            "coordinate_minimum": -1.0,
            "coordinate_maximum": 1.0,
        },
        "condition_encoder": {
            "frequency_count": 463,
            "convolution_channels": 32,
            "direction_embedding_dimension": 128,
            "latent_dimension": latent_dimension,
        },
        "model": {
            "coordinate_dimension": 5,
            "hidden_width": 256,
            "sine_layer_count": 6,
            "first_omega": 20.0,
            "hidden_omega": 20.0,
            "output_dimension": 2,
            "latent_dimension": latent_dimension,
            "modulation_variant": "phase",
            "placement": "hidden",
            "modulator_width": 128,
            "gamma_max_delta": 0.5,
            "beta_max": 3.141592653589793,
            "amplitude_max_delta": 0.5,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
        },
        "objective": "normalized_residual_mse",
        "cycles": 100,
        "steps_per_cycle": 262,
        "directions_per_step": 16,
        "validation_interval_cycles": 5,
        "validation_directions_per_block": 32,
        "gradient_clip": 5.0,
        "automatic_mixed_precision": False,
        "precision": "FP32",
        "seed": 20260821,
        "require_cuda": True,
        "require_clean_git": True,
        "test_policy": "The entry point resolves only the 262 train and 44 validation rows from the frozen split CSV. Test paths are never constructed or opened.",
        "run_name": f"sonicom_film_siren_b_latent_{latent_dimension}_seed20260821",
    }


def main() -> None:
    root = project_root()
    output_dir = root / "configs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for latent_dimension in (64, 128, 256):
        path = output_dir / f"sonicom_film_siren_b_latent_{latent_dimension}.json"
        path.write_text(
            json.dumps(configuration(latent_dimension), indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.append(str(path))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
