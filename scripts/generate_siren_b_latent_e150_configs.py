"""Generate the six frozen 150-cycle latent budget-rerun configs."""

from __future__ import annotations

import json

from mcar.paths import project_root


def main() -> None:
    root = project_root()
    outputs: list[str] = []
    for latent_dimension in (128, 256):
        for seed in (20260821, 20260822, 20260823):
            if seed == 20260821:
                source = (
                    root
                    / "configs"
                    / "experiments"
                    / f"sonicom_film_siren_b_latent_{latent_dimension}.json"
                )
            else:
                source = (
                    root
                    / "configs"
                    / "experiments"
                    / f"sonicom_film_siren_b_latent_{latent_dimension}_seed{seed}.json"
                )
            configuration = json.loads(source.read_text(encoding="utf-8"))
            configuration["experiment_id"] = (
                f"SIREN-B-LATENT-{latent_dimension}-SEED-{seed}-E150"
            )
            configuration["model_version"] = (
                f"film-siren-b-latent-{latent_dimension}-seed-{seed}-e150-v1"
            )
            configuration["protocol_amendment"] = (
                "experiments/film_siren/STAGE_B_LATENT_BUDGET_AMENDMENT.md"
            )
            configuration["cycles"] = 150
            configuration["run_name"] = (
                f"sonicom_film_siren_b_latent_{latent_dimension}_seed{seed}_e150"
            )
            output = (
                root
                / "configs"
                / "experiments"
                / f"sonicom_film_siren_b_latent_{latent_dimension}_seed{seed}_e150.json"
            )
            output.write_text(
                json.dumps(configuration, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append(str(output))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
