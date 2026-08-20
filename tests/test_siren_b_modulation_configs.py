from __future__ import annotations

import json

import pytest

from mcar.paths import project_root


ROOT = project_root()


@pytest.mark.parametrize("variant", ["concat", "amplitude", "phase", "full"])
def test_modulation_screening_configs_match_frozen_protocol(variant: str) -> None:
    path = (
        ROOT
        / "configs"
        / "experiments"
        / f"sonicom_film_siren_b_modulation_{variant}_seed20260821.json"
    )
    configuration = json.loads(path.read_text(encoding="utf-8"))
    assert configuration["search_stage"] == "modulation_screening"
    assert configuration["condition_encoder"]["latent_dimension"] == 128
    assert configuration["model"]["latent_dimension"] == 128
    assert configuration["model"]["modulation_variant"] == variant
    assert configuration["model"]["placement"] == (
        "none" if variant == "concat" else "hidden"
    )
    assert configuration["cycles"] == 100
    assert configuration["steps_per_cycle"] == 262
    assert configuration["seed"] == 20260821
    assert configuration["require_clean_git"] is True
