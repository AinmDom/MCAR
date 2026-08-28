"""Generate the two matched-seed Stage-D D1 extension configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "sonicom_film_siren_spectral_cnn_d1_seed20260821_e40.json"
)
MEMBERS = {
    20260822: {
        "checkpoint": (
            "artifacts/training/"
            "sonicom_film_siren_gl_final_d1d2_notch_seed20260822_e130/last.pt"
        ),
        "sha256": "5463CFE9FA618F5CEA0ADF7415E7CEAC978D071C24FB2038772A968519808A1D",
    },
    20260823: {
        "checkpoint": (
            "artifacts/training/"
            "sonicom_film_siren_gl_final_d1d2_notch_seed20260823_e130/last.pt"
        ),
        "sha256": "DD3C55F6E7FC32A5AC089272978BEF771009BF140768E19A33EDD74E1C704070",
    },
}


def main() -> None:
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    for seed, member in MEMBERS.items():
        configuration = copy.deepcopy(base)
        configuration["experiment_id"] = (
            f"FILM-SIREN-SPECTRAL-CNN-D1-SEED{seed}-E40"
        )
        configuration["initial_film_checkpoint"] = member["checkpoint"]
        configuration["initial_film_checkpoint_sha256"] = member["sha256"]
        configuration["seed"] = seed
        configuration["run_name"] = (
            f"sonicom_film_siren_spectral_cnn_d1_seed{seed}_e40"
        )
        output_path = (
            ROOT
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_spectral_cnn_d1_seed{seed}_e40.json"
        )
        output_path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
