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
    20260821: {
        "checkpoint": (
            "artifacts/training/"
            "sonicom_film_siren_gl_final_d1d2_notch_seed20260821_e130/last.pt"
        ),
        "sha256": "E37676D843D608B4DDD7B311EBA8B28A933183A03E7BAA49E1CF20E35F8B4129",
    },
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
        if seed == 20260821:
            continue
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

    for seed, member in MEMBERS.items():
        configuration = copy.deepcopy(base)
        configuration["experiment_id"] = (
            f"FILM-SIREN-SPECTRAL-CNN-FINAL-SEED{seed}-E190"
        )
        configuration["search_stage"] = "formal_fixed_cycle_e190"
        configuration["initial_film_checkpoint"] = member["checkpoint"]
        configuration["initial_film_checkpoint_sha256"] = member["sha256"]
        configuration["cycles"] = 190
        configuration["scheduler"]["horizon_cycles"] = 200
        configuration["seed"] = seed
        configuration["run_name"] = (
            f"sonicom_film_siren_spectral_cnn_final_seed{seed}_e190"
        )
        configuration["formal_fixed_cycle"] = True
        configuration["checkpoint_policy"] = "fixed_stop_cycle_last"
        output_path = (
            ROOT
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_spectral_cnn_final_seed{seed}_e190.json"
        )
        output_path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(output_path.relative_to(ROOT))

    for seed, member in MEMBERS.items():
        configuration = copy.deepcopy(base)
        configuration["experiment_id"] = (
            f"FILM-SIREN-SPECTRAL-CNN-D2-SEED{seed}-E200"
        )
        configuration["search_stage"] = "d2_e200_best_cycle_search"
        configuration["initial_film_checkpoint"] = member["checkpoint"]
        configuration["initial_film_checkpoint_sha256"] = member["sha256"]
        configuration["cycles"] = 200
        configuration["scheduler"]["horizon_cycles"] = 200
        configuration["seed"] = seed
        configuration["run_name"] = (
            f"sonicom_film_siren_spectral_cnn_d2_seed{seed}_e200"
        )
        output_path = (
            ROOT
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_spectral_cnn_d2_seed{seed}_e200.json"
        )
        output_path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
