"""Generate the four frozen Stage A4 confirmation configuration files.

Cell specifications (STAGE_A4_CONFIRMATION_PROTOCOL.md section 2):

- c1: D-o20-d6-w256-ho20 (main candidate)
- c2: linear-o30-d6-w256-ho30 (historical baseline / A2 L-o30)
- c3: D-o30-d6-w256-ho30 (A2 second-place control)
- c4: D-o20-d4-w256-ho20 (A3 depth tight-second control)

All share: 32 confirmation subjects, A2 64-direction holdout, 250 epochs,
seed 20260820. Deterministic generator.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcar.paths import project_root

SEED = 20260820
EPOCHS = 250

CELLS = {
    "c1": {
        "run_name": "sonicom_siren_a4_c1",
        "experiment_id": "SIREN-A4-C1",
        "model_version": "sonicom-siren-a4-c1",
        "mode": "dual", "first_omega": 20.0, "depth": 6,
        "width": 256, "hidden_omega": 20.0,
    },
    "c2": {
        "run_name": "sonicom_siren_a4_c2",
        "experiment_id": "SIREN-A4-C2",
        "model_version": "sonicom-siren-a4-c2",
        "mode": "linear", "first_omega": 30.0, "depth": 6,
        "width": 256, "hidden_omega": 30.0,
    },
    "c3": {
        "run_name": "sonicom_siren_a4_c3",
        "experiment_id": "SIREN-A4-C3",
        "model_version": "sonicom-siren-a4-c3",
        "mode": "dual", "first_omega": 30.0, "depth": 6,
        "width": 256, "hidden_omega": 30.0,
    },
    "c4": {
        "run_name": "sonicom_siren_a4_c4",
        "experiment_id": "SIREN-A4-C4",
        "model_version": "sonicom-siren-a4-c4",
        "mode": "dual", "first_omega": 20.0, "depth": 4,
        "width": 256, "hidden_omega": 20.0,
    },
}


def main() -> None:
    root = project_root()
    output_dir = root / "configs" / "experiments"
    for cell_id, spec in CELLS.items():
        configuration = {
            "schema_version": "1.0",
            "created_on": "2026-08-20",
            "experiment_id": spec["experiment_id"],
            "model_version": spec["model_version"],
            "experiment_type": "multi_subject_siren_matrix",
            "purpose": (
                f"Stage A4 confirmation cell {cell_id}; 32 unseen train "
                "subjects, 64-direction holdout. Frozen by "
                "STAGE_A4_CONFIRMATION_PROTOCOL.md."
            ),
            "dataset_root": "data/processed/sonicom_residual_q26_v1",
            "subject_split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "subjects_csv": "configs/data/siren_a4_confirmation_subjects_v1.csv",
            "holdout_csv": "configs/data/siren_a2_holdout_v1.csv",
            "interpolation_only": False,
            "direction_protocol": "train_on_793_minus_64_holdout_evaluate_holdout_64",
            "frequency_mapping": {
                "mode": spec["mode"],
                "formula_version": (
                    "dual_linear_erb_v1" if spec["mode"] == "dual"
                    else "linear_minmax_v1"
                ),
                "frequency_minimum_hz": 86.1328125,
                "frequency_maximum_hz": 19982.8125,
                "coordinate_minimum": -1.0,
                "coordinate_maximum": 1.0,
            },
            "model": {
                "hidden_width": spec["width"],
                "sine_layer_count": spec["depth"],
                "first_omega": spec["first_omega"],
                "hidden_omega": float(spec["hidden_omega"]),
                "output_dimension": 2,
                "output_unit": "normalized_residual",
            },
            "objective": "normalized_residual_mse",
            "optimizer": {
                "name": "Adam",
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
            },
            "epochs": EPOCHS,
            "steps_per_epoch": 100,
            "directions_per_batch": 16,
            "evaluation_directions_per_block": 32,
            "gradient_clip": 5.0,
            "automatic_mixed_precision": False,
            "precision": "FP32",
            "seed": SEED,
            "inference_repetitions": 5,
            "test_policy": (
                "Subjects resolve through "
                "configs/data/siren_a4_confirmation_subjects_v1.csv and the "
                "frozen split CSV; only train HDF5 files are opened. "
                "Validation and test HDF5 files are never read."
            ),
            "run_name": spec["run_name"],
        }
        path = output_dir / f"{spec['run_name']}.json"
        path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(path.name)


if __name__ == "__main__":
    main()
