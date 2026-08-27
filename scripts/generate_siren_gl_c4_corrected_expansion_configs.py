"""Generate the two prospective D1/D2+notch seed-expansion configs."""

from __future__ import annotations

import json
from copy import deepcopy

from mcar.paths import project_root


SOURCE = (
    "configs/experiments/"
    "sonicom_film_siren_gl_c4_d1d2_notch_seed20260821_e150.json"
)
AMENDMENT = (
    "experiments/film_siren/"
    "STAGE_C_GLOBAL_LOCAL_MCA_C4_COMMON_SCORE_CORRECTION.md"
)
SEEDS = (20260822, 20260823)


def main() -> None:
    root = project_root()
    base = json.loads((root / SOURCE).read_text(encoding="utf-8"))
    expected_objective = (0.25, 0.15, 0.30)
    objective = base["objective"]
    actual_objective = (
        objective["high_frequency_first_difference_weight"],
        objective["high_frequency_second_difference_weight"],
        objective["notch_depth_weight"],
    )
    if actual_objective != expected_objective:
        raise SystemExit(f"source objective mismatch: {actual_objective}")
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

    output_root = root / "configs" / "experiments"
    for seed in SEEDS:
        configuration = deepcopy(base)
        run_name = (
            "sonicom_film_siren_gl_c4_corrected_d1d2_notch_"
            f"seed{seed}_e150"
        )
        configuration.update(
            {
                "created_on": "2026-08-27",
                "experiment_id": (
                    f"SIREN-GL-C4-CORRECTED-D1D2-NOTCH-SEED{seed}-E150"
                ),
                "model_version": (
                    f"film-siren-gl-c4-corrected-d1d2-notch-seed{seed}-v1"
                ),
                "search_stage": "global_local_c4_corrected_seed_expansion",
                "seed": seed,
                "run_name": run_name,
                "execution_amendment": AMENDMENT,
                "cross_objective_selection_metric": "standardized_c0",
                "within_candidate_checkpoint_metric": (
                    "augmented_d1d2_notch_objective_total"
                ),
            }
        )
        path = output_root / f"{run_name}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
