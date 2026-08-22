"""Generate the frozen Stage B placement ranking-reversal RETEST configs."""

from __future__ import annotations

import copy
import json

from mcar.paths import project_root


PLACEMENTS = ("hidden", "all")
SEEDS = (20260824, 20260825)
AMENDMENT = "experiments/film_siren/STAGE_B_PLACEMENT_RETEST_PROTOCOL.md"


def main() -> None:
    root = project_root()
    decision_path = (
        root / "results" / "sonicom_film_siren_b_placement" / "decision.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not (
        decision.get("three_seed_complete") is True
        and decision.get("seed_ranking_reversed") is True
        and decision.get("winner") is None
        and set(decision.get("screening_top2", [])) == set(PLACEMENTS)
    ):
        raise ValueError(f"Placement ranking-reversal RETEST is not authorized: {decision}")

    outputs: list[str] = []
    for placement in PLACEMENTS:
        base_path = (
            root
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_b_placement_{placement}_seed20260821.json"
        )
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            configuration = copy.deepcopy(base)
            configuration["experiment_id"] = (
                f"SIREN-B-PLACEMENT-{placement.upper()}-RETEST-SEED-{seed}"
            )
            configuration["model_version"] = (
                f"film-siren-b-placement-{placement}-retest-seed-{seed}-v1"
            )
            configuration["search_stage"] = "placement_retest"
            configuration["protocol_amendment"] = AMENDMENT
            configuration["seed"] = seed
            configuration["run_name"] = (
                f"sonicom_film_siren_b_placement_{placement}_seed{seed}_retest"
            )
            output = (
                root
                / "configs"
                / "experiments"
                / f"sonicom_film_siren_b_placement_{placement}_seed{seed}_retest.json"
            )
            output.write_text(
                json.dumps(configuration, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append(str(output))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
