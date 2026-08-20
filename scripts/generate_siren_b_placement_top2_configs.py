"""Generate seed follow-ups for the frozen Stage B placement Top-2."""

from __future__ import annotations

import copy
import json

from mcar.paths import project_root


SEEDS = (20260822, 20260823)


def main() -> None:
    root = project_root()
    decision = json.loads(
        (
            root / "results" / "sonicom_film_siren_b_placement" / "decision.json"
        ).read_text(encoding="utf-8")
    )
    top2 = [str(value) for value in decision["screening_top2"]]
    if (
        len(top2) != 2
        or decision["screening_complete"] is not True
        or decision["budget_retest_required"] is True
    ):
        raise ValueError(f"Placement Top-2 is not frozen: {decision}")
    outputs: list[str] = []
    for placement in top2:
        base = json.loads(
            (
                root
                / "configs"
                / "experiments"
                / f"sonicom_film_siren_b_placement_{placement}_seed20260821.json"
            ).read_text(encoding="utf-8")
        )
        for seed in SEEDS:
            configuration = copy.deepcopy(base)
            configuration["experiment_id"] = (
                f"SIREN-B-PLACEMENT-{placement.upper()}-SEED-{seed}"
            )
            configuration["model_version"] = (
                f"film-siren-b-placement-{placement}-seed-{seed}-v1"
            )
            configuration["seed"] = seed
            configuration["run_name"] = (
                f"sonicom_film_siren_b_placement_{placement}_seed{seed}"
            )
            output = (
                root
                / "configs"
                / "experiments"
                / f"sonicom_film_siren_b_placement_{placement}_seed{seed}.json"
            )
            output.write_text(
                json.dumps(configuration, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append(str(output))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
