"""Generate the frozen Stage B placement-screening configurations."""

from __future__ import annotations

import copy
import json

from mcar.paths import project_root


PLACEMENTS = ("late", "hidden", "all")
SEED = 20260821


def main() -> None:
    root = project_root()
    decision = json.loads(
        (
            root
            / "results"
            / "sonicom_film_siren_b_modulation"
            / "decision.json"
        ).read_text(encoding="utf-8")
    )
    if decision["winner"] != "full" or decision["skip_placement"]:
        raise ValueError(f"Full modulation winner is not frozen: {decision}")
    base = json.loads(
        (
            root
            / "configs"
            / "experiments"
            / "sonicom_film_siren_b_modulation_full_seed20260821.json"
        ).read_text(encoding="utf-8")
    )
    outputs: list[str] = []
    for placement in PLACEMENTS:
        configuration = copy.deepcopy(base)
        configuration["experiment_id"] = f"SIREN-B-PLACEMENT-{placement.upper()}"
        configuration["model_version"] = f"film-siren-b-placement-{placement}-v1"
        configuration["search_stage"] = "placement_screening"
        configuration["model"]["placement"] = placement
        configuration["seed"] = SEED
        configuration["run_name"] = (
            f"sonicom_film_siren_b_placement_{placement}_seed{SEED}"
        )
        path = (
            root
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_b_placement_{placement}_seed{SEED}.json"
        )
        path.write_text(
            json.dumps(configuration, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.append(str(path))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
