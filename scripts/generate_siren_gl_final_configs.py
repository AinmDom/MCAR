"""Generate the three frozen E140 formal-training configurations."""

from __future__ import annotations

import json
from copy import deepcopy

from mcar.paths import project_root


SELECTION = "results/sonicom_film_siren_gl_top3/final_selection.json"
FREEZE = (
    "experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_FINAL_TRAINING_FREEZE.md"
)


def main() -> None:
    root = project_root()
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    if selection["winner"] != "cosine_c0" or int(selection["e_final"]) != 140:
        raise SystemExit("final selection must be cosine_c0 with E_final=140")
    source_path = root / selection["winner_source_config"]
    base = json.loads(source_path.read_text(encoding="utf-8"))
    if base["scheduler"] != {
        "name": "cosine",
        "horizon_cycles": 150,
        "warmup_cycles": 0,
    }:
        raise SystemExit("winner source scheduler mismatch")
    if base["optimizer"] != {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    }:
        raise SystemExit("winner source optimizer mismatch")
    config_root = root / "configs" / "experiments"
    for member_index, seed in enumerate(selection["formal_seeds"], start=1):
        configuration = deepcopy(base)
        run_name = f"sonicom_film_siren_gl_final_cosine_c0_seed{seed}_e140"
        configuration.update(
            {
                "created_on": "2026-08-27",
                "experiment_id": f"SIREN-GL-FINAL-COSINE-C0-SEED{seed}-E140",
                "model_version": f"film-siren-gl-final-cosine-c0-seed{seed}-v1",
                "search_stage": "formal_fixed_cycle_ensemble_member",
                "seed": seed,
                "cycles": 140,
                "run_name": run_name,
                "execution_amendment": FREEZE,
                "final_selection": SELECTION,
                "formal_fixed_cycle": True,
                "checkpoint_policy": "fixed_stop_cycle_last",
                "ensemble": {
                    "type": "residual_db_mean",
                    "member_index": member_index,
                    "member_count": 3,
                    "weight": 1 / 3,
                },
            }
        )
        path = config_root / f"{run_name}.json"
        path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
