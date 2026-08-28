"""Generate the three fixed-cycle Stage-E formal training configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from mcar.paths import project_root


SEEDS = (20260821, 20260822, 20260823)


def source_config(root: Path, seed: int) -> Path:
    stage = "d1" if seed == 20260821 else "d2"
    return root / "configs/experiments" / (
        f"sonicom_bounded_mcar_film_correction_{stage}_seed{seed}_e40.json"
    )


def main() -> None:
    root = project_root()
    for seed in SEEDS:
        config = copy.deepcopy(
            json.loads(source_config(root, seed).read_text(encoding="utf-8"))
        )
        config["experiment_id"] = (
            f"BOUNDED-MCAR-FILM-CORRECTION-FORMAL-SEED{seed}-E25"
        )
        config["protocol"] = (
            "experiments/film_siren/"
            "STAGE_E_BOUNDED_CORRECTION_FORMAL_E25_FREEZE.md"
        )
        config["search_stage"] = "formal_fixed_e25_three_member_ensemble"
        config["cycles"] = 25
        config["run_name"] = (
            f"sonicom_bounded_mcar_film_correction_final_seed{seed}_e25"
        )
        config["formal_fixed_cycle"] = True
        config["checkpoint_policy"] = "fixed_stop_cycle_last"
        output = root / "configs/experiments" / (
            f"sonicom_bounded_mcar_film_correction_final_seed{seed}_e25.json"
        )
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(output.relative_to(root))


if __name__ == "__main__":
    main()
