from __future__ import annotations

import json

import pytest

from mcar.paths import project_root
import numpy as np

from scripts.analyze_siren_b_placement_retest import (
    exact_paired_bootstrap_interval,
    select_retest_winner,
)


ROOT = project_root()
AMENDMENT = "experiments/film_siren/STAGE_B_PLACEMENT_RETEST_PROTOCOL.md"


@pytest.mark.parametrize("placement", ["hidden", "all"])
@pytest.mark.parametrize("seed", [20260824, 20260825])
def test_placement_retest_configs_change_only_authorized_fields(
    placement: str, seed: int
) -> None:
    base_path = (
        ROOT
        / "configs"
        / "experiments"
        / f"sonicom_film_siren_b_placement_{placement}_seed20260821.json"
    )
    retest_path = (
        ROOT
        / "configs"
        / "experiments"
        / f"sonicom_film_siren_b_placement_{placement}_seed{seed}_retest.json"
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    retest = json.loads(retest_path.read_text(encoding="utf-8"))
    for key in (
        "dataset_root",
        "subject_split_csv",
        "q26_csv",
        "q26_normalization",
        "direction_protocol",
        "frequency_mapping",
        "condition_encoder",
        "model",
        "optimizer",
        "objective",
        "cycles",
        "steps_per_cycle",
        "directions_per_step",
        "validation_interval_cycles",
        "validation_directions_per_block",
        "gradient_clip",
        "automatic_mixed_precision",
        "precision",
        "require_cuda",
        "require_clean_git",
        "test_policy",
    ):
        assert retest[key] == base[key]
    assert retest["search_stage"] == "placement_retest"
    assert retest["protocol_amendment"] == AMENDMENT
    assert retest["seed"] == seed
    assert retest["run_name"] == (
        f"sonicom_film_siren_b_placement_{placement}_seed{seed}_retest"
    )


def test_retest_winner_uses_lower_five_seed_mean() -> None:
    assert (
        select_retest_winner(
            {"hidden": 3.1, "all": 3.09}, complete=True, budget_retest=False
        )
        == "all"
    )
    assert (
        select_retest_winner(
            {"hidden": 3.08, "all": 3.09}, complete=True, budget_retest=False
        )
        == "hidden"
    )


@pytest.mark.parametrize(
    ("complete", "budget_retest"), [(False, False), (True, True)]
)
def test_retest_winner_is_blocked_when_incomplete_or_budget_retest(
    complete: bool, budget_retest: bool
) -> None:
    assert (
        select_retest_winner(
            {"hidden": 3.08, "all": 3.09},
            complete=complete,
            budget_retest=budget_retest,
        )
        is None
    )


def test_retest_exact_tie_has_no_winner() -> None:
    assert (
        select_retest_winner(
            {"hidden": 3.09, "all": 3.09}, complete=True, budget_retest=False
        )
        is None
    )


def test_exact_paired_bootstrap_interval_is_deterministic() -> None:
    deltas = np.asarray([-0.01, -0.005, 0.0, 0.005, 0.01])
    first = exact_paired_bootstrap_interval(deltas)
    second = exact_paired_bootstrap_interval(deltas)
    assert first == second
    assert first[0] < 0.0 < first[1]
