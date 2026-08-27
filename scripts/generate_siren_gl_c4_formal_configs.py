"""Generate the three frozen E130 formal-training configurations (D1/D2+notch).

Mirrors scripts/generate_siren_gl_final_configs.py for the corrected C4
candidate: warmup-cosine + AdamW 1e-4/1e-4 + D1/D2/notch 0.25/0.15/0.30,
seven-dimensional global+local MCA, formal fixed-cycle with authoritative
last.pt at cycle 130.
"""

from __future__ import annotations

import json
from copy import deepcopy

from mcar.paths import project_root


SELECTION = "results/sonicom_film_siren_gl_c4_corrected_expansion/selection.json"
FREEZE = (
    "experiments/film_siren/"
    "STAGE_C_GLOBAL_LOCAL_MCA_C4_CORRECTED_FORMAL_TRAINING_FREEZE.md"
)
SOURCE = (
    "configs/experiments/"
    "sonicom_film_siren_gl_c4_d1d2_notch_seed20260821_e150.json"
)


def main() -> None:
    root = project_root()
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    if selection["candidate"] != "warmup_cosine_d1d2_notch":
        raise SystemExit("selection candidate must be warmup_cosine_d1d2_notch")
    if int(selection["e_final"]) != 130:
        raise SystemExit("selection e_final must be 130")
    if [int(s) for s in selection["formal_seeds"]] != [20260821, 20260822, 20260823]:
        raise SystemExit("selection formal_seeds mismatch")
    if int(selection["scheduler_horizon_cycles"]) != 150:
        raise SystemExit("selection scheduler horizon mismatch")
    if selection["formal_checkpoint_policy"] != "fixed_stop_cycle_last":
        raise SystemExit("selection checkpoint policy mismatch")

    base = json.loads((root / SOURCE).read_text(encoding="utf-8"))
    if base["scheduler"] != {
        "name": "warmup_cosine",
        "horizon_cycles": 150,
        "warmup_cycles": 5,
    }:
        raise SystemExit("source scheduler mismatch")
    if base["optimizer"] != {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    }:
        raise SystemExit("source optimizer mismatch")
    objective = base["objective"]
    actual = (
        objective["high_frequency_first_difference_weight"],
        objective["high_frequency_second_difference_weight"],
        objective["notch_depth_weight"],
    )
    if actual != (0.25, 0.15, 0.30):
        raise SystemExit(f"source objective mismatch: {actual}")

    config_root = root / "configs" / "experiments"
    for member_index, seed in enumerate(selection["formal_seeds"], start=1):
        configuration = deepcopy(base)
        run_name = f"sonicom_film_siren_gl_final_d1d2_notch_seed{seed}_e130"
        configuration.update(
            {
                "created_on": "2026-08-27",
                "experiment_id": f"SIREN-GL-FINAL-D1D2-NOTCH-SEED{seed}-E130",
                "model_version": f"film-siren-gl-final-d1d2-notch-seed{seed}-v1",
                "search_stage": "formal_fixed_cycle_ensemble_member",
                "seed": seed,
                "cycles": 130,
                "run_name": run_name,
                "execution_amendment": FREEZE,
                "final_selection": SELECTION,
                "formal_fixed_cycle": True,
                "checkpoint_policy": "fixed_stop_cycle_last",
                "cross_objective_selection_metric": "standardized_c0",
                "within_candidate_checkpoint_metric": (
                    "augmented_d1d2_notch_objective_total"
                ),
                "ensemble": {
                    "type": "residual_db_mean",
                    "member_index": member_index,
                    "member_count": 3,
                    "weight": 1 / 3,
                },
            }
        )
        path = config_root / f"{run_name}.json"
        path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
