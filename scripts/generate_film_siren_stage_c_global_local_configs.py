"""Generate pre-registered global+local Stage C configuration sets."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from mcar.paths import project_root


PROTOCOL_AMENDMENT = (
    "experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_REVALIDATION_PROTOCOL.md"
)


def main() -> None:
    root = project_root()
    config_root = root / "configs" / "experiments"
    source = json.loads(
        (
            config_root
            / "sonicom_film_siren_c1_lr1e4_adam_seed20260821_e150.json"
        ).read_text(encoding="utf-8")
    )
    candidates = (("lr3e5", 3e-5), ("lr1e4", 1e-4), ("lr3e4", 3e-4))
    for suffix, learning_rate in candidates:
        configuration = deepcopy(source)
        slug = suffix.upper()
        configuration.update(
            {
                "created_on": "2026-08-26",
                "experiment_id": f"SIREN-GL-C1-{slug}-ADAM-SEED-20260821-E150",
                "model_version": (
                    f"film-siren-gl-c1-c0-adam-{suffix}-seed20260821-v1"
                ),
                "protocol_amendment": PROTOCOL_AMENDMENT,
                "search_stage": "global_local_c1_learning_rate_revalidation",
                "conditioning_scope": "global_plus_local_mca",
                "local_mca_policy": (
                    "append_binaural_query_mca_logmag_db_normalized_by_"
                    "train_only_statistics"
                ),
                "run_name": (
                    f"sonicom_film_siren_gl_c1_{suffix}_adam_"
                    "seed20260821_e150"
                ),
            }
        )
        configuration["model"]["coordinate_dimension"] = 7
        configuration["optimizer"]["learning_rate"] = learning_rate
        path = config_root / f"sonicom_film_siren_gl_c1_{suffix}_adam_seed20260821_e150.json"
        path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(root))

    c1_winner = json.loads(
        (
            config_root
            / "sonicom_film_siren_gl_c1_lr1e4_adam_seed20260821_e150.json"
        ).read_text(encoding="utf-8")
    )
    c2_amendment = (
        "experiments/film_siren/"
        "STAGE_C_GLOBAL_LOCAL_MCA_C2_PARALLEL_AMENDMENT.md"
    )
    c2_candidates = (
        ("adam_wd0", "Adam", 0.0),
        ("adamw_wd1e5", "AdamW", 1e-5),
        ("adamw_wd1e4", "AdamW", 1e-4),
    )
    for suffix, optimizer, weight_decay in c2_candidates:
        configuration = deepcopy(c1_winner)
        configuration.update(
            {
                "experiment_id": (
                    f"SIREN-GL-C2-{suffix.upper()}-SEED-20260821-E150"
                ),
                "model_version": f"film-siren-gl-c2-{suffix}-seed20260821-v1",
                "execution_amendment": c2_amendment,
                "search_stage": "global_local_c2_optimizer_weight_decay",
                "run_name": (
                    f"sonicom_film_siren_gl_c2_{suffix}_seed20260821_e150"
                ),
            }
        )
        configuration["optimizer"] = {
            "name": optimizer,
            "learning_rate": 1e-4,
            "weight_decay": weight_decay,
        }
        path = config_root / (
            f"sonicom_film_siren_gl_c2_{suffix}_seed20260821_e150.json"
        )
        path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
