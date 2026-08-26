"""Generate the seven new configs for the frozen global/local-MCA expansion."""

from __future__ import annotations

import json
from copy import deepcopy

from mcar.paths import project_root


PROTOCOL = "experiments/film_siren/STAGE_B_GLOBAL_LOCAL_MCA_EXPANSION_PROTOCOL.md"
SCOPES = ("global_zero_local", "local_mca", "global_plus_local_mca")
SEEDS = (20260821, 20260822, 20260823)


def main() -> None:
    root = project_root()
    config_dir = root / "configs" / "experiments"
    templates = {
        "global_zero_local": json.loads(
            (
                config_dir
                / "sonicom_film_siren_b_gate_global_zero_local_seed20260821_e150.json"
            ).read_text(encoding="utf-8")
        ),
        "global_plus_local_mca": json.loads(
            (
                config_dir
                / "sonicom_film_siren_b_gate_global_plus_local_mca_seed20260821_e150.json"
            ).read_text(encoding="utf-8")
        ),
    }
    templates["local_mca"] = deepcopy(templates["global_plus_local_mca"])
    for scope in SCOPES:
        for seed in SEEDS:
            if seed == 20260821 and scope != "local_mca":
                continue
            configuration = deepcopy(templates[scope])
            slug = scope.replace("_", "-").upper()
            run_name = f"sonicom_film_siren_b_expansion_{scope}_seed{seed}_e150"
            configuration.update(
                {
                    "created_on": "2026-08-26",
                    "experiment_id": f"SIREN-B-GLOBAL-LOCAL-EXPANSION-{slug}-SEED-{seed}",
                    "model_version": f"film-siren-b-expansion-{scope.replace('_', '-')}-e150-seed-{seed}-v1",
                    "protocol_amendment": PROTOCOL,
                    "search_stage": "global_local_mca_expansion",
                    "conditioning_scope": scope,
                    "seed": seed,
                    "run_name": run_name,
                }
            )
            configuration["local_mca_policy"] = {
                "global_zero_local": "append_two_zero_channels_without_reading_mca_logmag_db",
                "local_mca": "append_normalized_binaural_query_mca_and_bypass_q26_encoder_with_zero_latent",
                "global_plus_local_mca": "append_binaural_query_mca_logmag_db_normalized_by_train_only_statistics",
            }[scope]
            path = config_dir / f"{run_name}.json"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
            print(path.relative_to(root))


if __name__ == "__main__":
    main()
