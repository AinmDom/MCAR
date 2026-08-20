"""Generate the frozen Stage B modulation-screening configurations."""

from __future__ import annotations

import copy
import json

from mcar.paths import project_root


VARIANTS = ("concat", "amplitude", "phase", "full")
SEED = 20260821


def main() -> None:
    root = project_root()
    decision = json.loads(
        (
            root
            / "results"
            / "sonicom_film_siren_b_latent"
            / "e150_decision.json"
        ).read_text(encoding="utf-8")
    )
    if decision["winner"] != 128 or not decision["three_seed_complete"]:
        raise ValueError(f"Latent winner is not frozen at 128: {decision}")
    base = json.loads(
        (
            root
            / "configs"
            / "experiments"
            / "sonicom_film_siren_b_latent_128.json"
        ).read_text(encoding="utf-8")
    )
    outputs: list[str] = []
    for variant in VARIANTS:
        configuration = copy.deepcopy(base)
        configuration["experiment_id"] = f"SIREN-B-MODULATION-{variant.upper()}"
        configuration["model_version"] = f"film-siren-b-modulation-{variant}-v1"
        configuration["search_stage"] = "modulation_screening"
        configuration["model"]["modulation_variant"] = variant
        configuration["model"]["placement"] = (
            "none" if variant == "concat" else "hidden"
        )
        configuration["cycles"] = 100
        configuration["seed"] = SEED
        configuration["run_name"] = (
            f"sonicom_film_siren_b_modulation_{variant}_seed{SEED}"
        )
        path = (
            root
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_b_modulation_{variant}_seed{SEED}.json"
        )
        path.write_text(
            json.dumps(configuration, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.append(str(path))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
