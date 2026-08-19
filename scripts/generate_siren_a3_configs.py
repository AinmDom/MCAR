"""Generate frozen Stage A3 configuration files.

Stage-aware generator:

- ``--stage budget --epochs 200``: D-o20 baseline budget-plateau run
  (run name ``sonicom_siren_a3_budget_e200``);
- ``--stage depth --epochs N``: three depth candidates
  (``sonicom_siren_a3_d{4,6,8}``, width 256, hidden omega 30);
- ``--stage width --epochs N --best-depth D``: three width candidates
  (``sonicom_siren_a3_d{D}_w{128,256,512}``);
- ``--stage ho --epochs N --best-depth D --best-width W``: three
  hidden-omega candidates (``sonicom_siren_a3_d{D}_w{W}_ho{20,30,50}``).

All configurations share the frozen D-o20 context: dual frequency mapping,
first omega 20, A2 subjects/holdout locks. The generator is deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcar.paths import project_root

SEED = 20260819
FIRST_OMEGA = 20.0
HIDDEN_OMEGA = 30.0
DEPTH_CANDIDATES = (4, 6, 8)
WIDTH_CANDIDATES = (128, 256, 512)
HIDDEN_OMEGA_CANDIDATES = (20.0, 30.0, 50.0)


def base_config(
    run_name: str,
    experiment_id: str,
    model_version: str,
    epochs: int,
    hidden_width: int,
    sine_layer_count: int,
    hidden_omega: float,
) -> dict:
    return {
        "schema_version": "1.0",
        "created_on": "2026-08-19",
        "experiment_id": experiment_id,
        "model_version": model_version,
        "experiment_type": "multi_subject_siren_matrix",
        "purpose": (
            "Stage A3 backbone deepening cell on the frozen D-o20 context; "
            "five fixed A2 train subjects, 64-direction holdout evaluation. "
            "Frozen by STAGE_A3_PROTOCOL.md."
        ),
        "dataset_root": "data/processed/sonicom_residual_q26_v1",
        "subject_split_csv": "configs/data/sonicom_subject_split_v1.csv",
        "subjects_csv": "configs/data/siren_a2_subjects_v1.csv",
        "holdout_csv": "configs/data/siren_a2_holdout_v1.csv",
        "interpolation_only": False,
        "direction_protocol": "train_on_793_minus_64_holdout_evaluate_holdout_64",
        "frequency_mapping": {
            "mode": "dual",
            "formula_version": "dual_linear_erb_v1",
            "frequency_minimum_hz": 86.1328125,
            "frequency_maximum_hz": 19982.8125,
            "coordinate_minimum": -1.0,
            "coordinate_maximum": 1.0,
        },
        "model": {
            "hidden_width": hidden_width,
            "sine_layer_count": sine_layer_count,
            "first_omega": FIRST_OMEGA,
            "hidden_omega": float(hidden_omega),
            "output_dimension": 2,
            "output_unit": "normalized_residual",
        },
        "objective": "normalized_residual_mse",
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
        },
        "epochs": epochs,
        "steps_per_epoch": 100,
        "directions_per_batch": 16,
        "evaluation_directions_per_block": 32,
        "gradient_clip": 5.0,
        "automatic_mixed_precision": False,
        "precision": "FP32",
        "seed": SEED,
        "inference_repetitions": 5,
        "test_policy": (
            "Subjects resolve through configs/data/siren_a2_subjects_v1.csv "
            "and the frozen split CSV; only train HDF5 files are opened. "
            "Validation and test HDF5 files are never read."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("budget", "depth", "width", "ho"))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--best-depth", type=int, default=6)
    parser.add_argument("--best-width", type=int, default=256)
    args = parser.parse_args()

    root = project_root()
    output_dir = root / "configs" / "experiments"
    if args.stage == "budget":
        cells = [
            ("sonicom_siren_a3_budget_e200", "SIREN-A3-BUDGET-E200",
             "sonicom-siren-a3-budget-e200", 256, 6, HIDDEN_OMEGA)
        ]
    elif args.stage == "depth":
        cells = [
            (f"sonicom_siren_a3_d{d}", f"SIREN-A3-DEPTH-D{d}",
             f"sonicom-siren-a3-depth-d{d}", 256, d, HIDDEN_OMEGA)
            for d in DEPTH_CANDIDATES
        ]
    elif args.stage == "width":
        cells = [
            (f"sonicom_siren_a3_d{args.best_depth}_w{w}",
             f"SIREN-A3-WIDTH-W{w}",
             f"sonicom-siren-a3-width-w{w}", w, args.best_depth, HIDDEN_OMEGA)
            for w in WIDTH_CANDIDATES
        ]
    else:  # ho
        cells = [
            (f"sonicom_siren_a3_d{args.best_depth}_w{args.best_width}_ho{int(ho)}",
             f"SIREN-A3-HO-O{int(ho)}",
             f"sonicom-siren-a3-ho-o{int(ho)}", args.best_width,
             args.best_depth, ho)
            for ho in HIDDEN_OMEGA_CANDIDATES
        ]
    for run_name, experiment_id, model_version, width, depth, ho in cells:
        configuration = base_config(
            run_name, experiment_id, model_version, args.epochs,
            width, depth, ho,
        )
        configuration["run_name"] = run_name
        path = output_dir / f"{run_name}.json"
        path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(path.name)


if __name__ == "__main__":
    main()
