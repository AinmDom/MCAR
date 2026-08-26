"""Generate the six Top-3 seed-expansion configurations.

Reads results/sonicom_film_siren_gl_top3/selection.json (produced by
analyze_siren_gl_top3.py), deduplicates the Top-3 by unique configuration
identity (optimizer + scheduler + objective weights), and for each unique
configuration generates two from-scratch configs with seeds 20260822 and
20260823. All other fields (base optimizer/scheduler/objective, data, sampling)
are deep-copied from the winning screening config so the expanded runs share
the exact search-stage configuration except for seed and identity metadata.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from mcar.paths import project_root

NEW_SEEDS = (20260822, 20260823)
TOP3_AMENDMENT = (
    "experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_TOP3_SEED_EXPANSION_AMENDMENT.md"
)
SELECTION_JSON = "results/sonicom_film_siren_gl_top3/selection.json"


def config_identity(configuration: dict[str, object]) -> tuple[object, ...]:
    objective = configuration["objective"]
    return (
        configuration["optimizer"]["name"],
        configuration["optimizer"]["learning_rate"],
        configuration["optimizer"]["weight_decay"],
        configuration["scheduler"]["name"],
        configuration["scheduler"]["horizon_cycles"],
        configuration["scheduler"]["warmup_cycles"],
        objective["erb_weight"],
        objective["high_frequency_weight"],
        objective["strict_ild_weight"],
        objective["spectral_band_ild_weight"],
        objective["high_frequency_first_difference_weight"],
        objective["high_frequency_second_difference_weight"],
        objective["notch_depth_weight"],
    )


def main() -> None:
    root = project_root()
    selection = json.loads((root / SELECTION_JSON).read_text(encoding="utf-8"))
    top3 = selection["top3"]
    if len(top3) != 3:
        raise SystemExit(f"expected 3 Top-3 entries, got {len(top3)}")

    config_root = root / "configs" / "experiments"
    unique: list[tuple[str, dict[str, object]]] = []
    seen: set[tuple[object, ...]] = set()
    for entry in top3:
        source_name = entry["run_name"]
        source_path = config_root / f"{source_name}.json"
        if not source_path.is_file():
            raise SystemExit(f"source config missing: {source_path.relative_to(root)}")
        configuration = json.loads(source_path.read_text(encoding="utf-8"))
        identity = config_identity(configuration)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((source_name, configuration))
    if len(unique) != 3:
        raise SystemExit(
            f"expected 3 unique Top-3 configurations, got {len(unique)} "
            "(deduplicated by optimizer+scheduler+objective)"
        )

    generated: list[str] = []
    for rank, (source_name, base) in enumerate(unique, start=1):
        source_variant = source_name.replace("sonicom_film_siren_", "").replace(
            "_seed20260821_e150", ""
        )
        for seed in NEW_SEEDS:
            configuration = deepcopy(base)
            run_name = f"sonicom_film_siren_gl_top3_rank{rank}_{source_variant}_seed{seed}_e150"
            configuration.update(
                {
                    "created_on": "2026-08-27",
                    "experiment_id": (
                        f"SIREN-GL-TOP3-RANK{rank}-SEED{seed}-E150"
                    ),
                    "model_version": f"film-siren-gl-top3-rank{rank}-seed{seed}-v1",
                    "execution_amendment": TOP3_AMENDMENT,
                    "search_stage": "global_top3_seed_expansion",
                    "seed": seed,
                    "run_name": run_name,
                }
            )
            path = config_root / f"{run_name}.json"
            path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
            generated.append(str(path.relative_to(root)))
            print(path.relative_to(root))


if __name__ == "__main__":
    main()
