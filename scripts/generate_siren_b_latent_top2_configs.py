"""Generate additional seeds for the frozen Stage B latent Top-2."""

from __future__ import annotations

import json

from mcar.paths import project_root


def main() -> None:
    root = project_root()
    decision_path = root / "results" / "sonicom_film_siren_b_latent" / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    top2 = [int(value) for value in decision["screening_top2"]]
    if top2 != [128, 256] or bool(decision["extend_384_512"]):
        raise ValueError(f"Unexpected frozen latent decision: {decision}")
    outputs: list[str] = []
    for latent_dimension in top2:
        base_path = (
            root
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_b_latent_{latent_dimension}.json"
        )
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for seed in (20260822, 20260823):
            configuration = dict(base)
            configuration["experiment_id"] = (
                f"SIREN-B-LATENT-{latent_dimension}-SEED-{seed}"
            )
            configuration["model_version"] = (
                f"film-siren-b-latent-{latent_dimension}-seed-{seed}-v1"
            )
            configuration["seed"] = seed
            configuration["run_name"] = (
                f"sonicom_film_siren_b_latent_{latent_dimension}_seed{seed}"
            )
            output = (
                root
                / "configs"
                / "experiments"
                / f"sonicom_film_siren_b_latent_{latent_dimension}_seed{seed}.json"
            )
            output.write_text(
                json.dumps(configuration, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append(str(output))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
