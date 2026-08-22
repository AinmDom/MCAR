from __future__ import annotations

import json

import pytest

from mcar.paths import project_root
from scripts.analyze_siren_b_conditioning_check_e150 import (
    plateau_decision,
    select_status,
)


ROOT = project_root()
AMENDMENT = "experiments/film_siren/STAGE_B_CONDITIONING_CHECK_E150_AMENDMENT.md"


@pytest.mark.parametrize("kind", ["conditioned", "unconditional"])
@pytest.mark.parametrize("seed", [20260821, 20260822, 20260823])
def test_e150_configs_only_change_budget_and_identity(kind: str, seed: int) -> None:
    if kind == "conditioned":
        base_name = f"sonicom_film_siren_b_placement_all_seed{seed}.json"
        e150_name = f"sonicom_film_siren_b_conditioning_check_all_seed{seed}_e150.json"
    else:
        base_name = f"sonicom_shared_siren_b_unconditional_seed{seed}.json"
        e150_name = f"sonicom_shared_siren_b_unconditional_seed{seed}_e150.json"
    directory = ROOT / "configs" / "experiments"
    base = json.loads((directory / base_name).read_text(encoding="utf-8"))
    e150 = json.loads((directory / e150_name).read_text(encoding="utf-8"))
    for key in (
        "dataset_root",
        "subject_split_csv",
        "direction_protocol",
        "frequency_mapping",
        "model",
        "optimizer",
        "objective",
        "steps_per_cycle",
        "directions_per_step",
        "validation_interval_cycles",
        "validation_directions_per_block",
        "gradient_clip",
        "precision",
        "seed",
        "require_cuda",
        "require_clean_git",
    ):
        assert e150[key] == base[key]
    if kind == "conditioned":
        assert e150["condition_encoder"] == base["condition_encoder"]
        assert e150["q26_csv"] == base["q26_csv"]
        assert e150["q26_normalization"] == base["q26_normalization"]
    else:
        assert "condition_encoder" not in e150
        assert "q26_csv" not in e150
        assert "q26_normalization" not in e150
    assert e150["cycles"] == 150
    assert e150["search_stage"] == "conditioning_check_e150"
    assert e150["protocol_amendment"] == AMENDMENT
    assert e150["run_name"].endswith("_e150")


def test_plateau_decision_keeps_early_best() -> None:
    assert plateau_decision(145, {}) == ("KEEP", None)


def test_plateau_decision_uses_window_threshold() -> None:
    flat = {cycle: 3.0 for cycle in range(100, 151, 5)}
    assert plateau_decision(150, flat) == ("KEEP_PLATEAU", 0.0)
    improving = {cycle: 3.0 for cycle in range(100, 130, 5)}
    improving.update({cycle: 2.99 for cycle in range(130, 151, 5)})
    decision, improvement = plateau_decision(150, improving)
    assert decision == "RETEST"
    assert improvement is not None and improvement >= 0.1


def test_status_requires_complete_sufficient_budget_and_improvement() -> None:
    assert select_status(3.0, 3.1, complete=True, budget_retest=False) == (
        True,
        "CONDITIONING_CHECK_PASSED",
    )
    assert select_status(3.2, 3.1, complete=True, budget_retest=False) == (
        False,
        "DO_NOT_FREEZE",
    )
    assert select_status(3.0, 3.1, complete=True, budget_retest=True) == (
        None,
        "PENDING",
    )
