"""Generate the two pre-registered Stage-E matched-seed E40 configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from mcar.paths import project_root


SOURCES = {
    20260822: (
        "artifacts/training/sonicom_film_siren_gl_final_d1d2_notch_seed20260822_e130/last.pt",
        "5463CFE9FA618F5CEA0ADF7415E7CEAC978D071C24FB2038772A968519808A1D",
    ),
    20260823: (
        "artifacts/training/sonicom_film_siren_gl_final_d1d2_notch_seed20260823_e130/last.pt",
        "DD3C55F6E7FC32A5AC089272978BEF771009BF140768E19A33EDD74E1C704070",
    ),
}


def main() -> None:
    root = project_root()
    base_path = root / "configs/experiments/sonicom_bounded_mcar_film_correction_d1_seed20260821_e40.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    for seed, (checkpoint, digest) in SOURCES.items():
        config = copy.deepcopy(base)
        config["experiment_id"] = f"BOUNDED-MCAR-FILM-CORRECTION-D2-SEED{seed}-E40"
        config["protocol"] = (
            "experiments/film_siren/"
            "STAGE_E_BOUNDED_CORRECTION_MATCHED_SEED_E40.md"
        )
        config["search_stage"] = "d2_matched_seed_stability_e40"
        config["seed"] = seed
        config["initial_film_checkpoint"] = checkpoint
        config["initial_film_checkpoint_sha256"] = digest
        config["run_name"] = (
            f"sonicom_bounded_mcar_film_correction_d2_seed{seed}_e40"
        )
        output = root / "configs/experiments" / f"sonicom_bounded_mcar_film_correction_d2_seed{seed}_e40.json"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(output.relative_to(root))


if __name__ == "__main__":
    main()
