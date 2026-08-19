"""Generate the nine frozen Stage A2 matrix configuration files.

Produces ``configs/experiments/sonicom_siren_a2_<mode>_w256_d6_o<omega>.json``
for ``mode in {linear, erb, dual}`` and ``first_omega in {20, 30, 50}``.
The generator is deterministic and committed so the lock files can be
reproduced exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcar.paths import project_root

MODES = ("linear", "erb", "dual")
OMEGAS = (20, 30, 50)
SEED = 20260819
FORMULA_VERSIONS = {
    "linear": "linear_minmax_v1",
    "erb": "erb_rate_v1",
    "dual": "dual_linear_erb_v1",
}


def build(mode: str, first_omega: int) -> dict:
    return {
        "schema_version": "1.0",
        "created_on": "2026-08-19",
        "experiment_id": f"SIREN-A2-{mode.upper()}-W256-D6-O{first_omega}",
        "model_version": f"sonicom-siren-a2-{mode}-o{first_omega}-v1",
        "experiment_type": "multi_subject_siren_matrix",
        "purpose": (
            "Stage A2 backbone matrix cell: five fixed train subjects, "
            "independent plain SIREN per subject, 64-direction holdout "
            "evaluation. Frozen by STAGE_A2_PROTOCOL.md."
        ),
        "dataset_root": "data/processed/sonicom_residual_q26_v1",
        "subject_split_csv": "configs/data/sonicom_subject_split_v1.csv",
        "subjects_csv": "configs/data/siren_a2_subjects_v1.csv",
        "holdout_csv": "configs/data/siren_a2_holdout_v1.csv",
        "interpolation_only": False,
        "direction_protocol": (
            "train_on_793_minus_64_holdout_evaluate_holdout_64"
        ),
        "frequency_mapping": {
            "mode": mode,
            "formula_version": FORMULA_VERSIONS[mode],
            "frequency_minimum_hz": 86.1328125,
            "frequency_maximum_hz": 19982.8125,
            "coordinate_minimum": -1.0,
            "coordinate_maximum": 1.0,
        },
        "model": {
            "hidden_width": 256,
            "sine_layer_count": 6,
            "first_omega": float(first_omega),
            "hidden_omega": 30.0,
            "output_dimension": 2,
            "output_unit": "normalized_residual",
        },
        "objective": "normalized_residual_mse",
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
        },
        "epochs": 100,
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
    root = project_root()
    output_dir = root / "configs" / "experiments"
    for mode in MODES:
        for first_omega in OMEGAS:
            configuration = build(mode, first_omega)
            run_name = f"sonicom_siren_a2_{mode}_w256_d6_o{first_omega}"
            configuration["run_name"] = run_name
            path = output_dir / f"{run_name}.json"
            path.write_text(
                json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
            )
            print(path.name)


if __name__ == "__main__":
    main()
