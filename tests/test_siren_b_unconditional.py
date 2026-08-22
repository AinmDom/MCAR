from __future__ import annotations

import json

import pytest
import torch

from mcar.models.siren import Siren, SirenConfig
from mcar.paths import project_root
from scripts.analyze_siren_b_unconditional import conditioning_passes


ROOT = project_root()


@pytest.mark.parametrize("seed", [20260821, 20260822, 20260823])
def test_unconditional_configs_match_frozen_baseline(seed: int) -> None:
    path = (
        ROOT
        / "configs"
        / "experiments"
        / f"sonicom_shared_siren_b_unconditional_seed{seed}.json"
    )
    configuration = json.loads(path.read_text(encoding="utf-8"))
    assert configuration["experiment_type"] == "joint_unconditional_siren_stage_b"
    assert configuration["model"] == {
        "input_dimension": 5,
        "hidden_width": 256,
        "sine_layer_count": 6,
        "first_omega": 20.0,
        "hidden_omega": 20.0,
        "output_dimension": 2,
    }
    assert configuration["cycles"] == 100
    assert configuration["steps_per_cycle"] == 262
    assert configuration["directions_per_step"] == 16
    assert configuration["seed"] == seed
    assert configuration["require_clean_git"] is True
    assert "q26_csv" not in configuration
    assert "q26_normalization" not in configuration
    assert "condition_encoder" not in configuration


def test_unconditional_model_forward_has_no_subject_input() -> None:
    model = Siren(
        SirenConfig(
            input_dimension=5,
            hidden_width=16,
            sine_layer_count=2,
            first_omega=20.0,
            hidden_omega=20.0,
            output_dimension=2,
        )
    )
    output = model(torch.randn(7, 5))
    assert output.shape == (7, 2)
    assert torch.isfinite(output).all()


def test_conditioning_check_requires_strict_improvement() -> None:
    assert conditioning_passes(3.0, 3.1, complete=True, budget_retest=False) is True
    assert conditioning_passes(3.1, 3.1, complete=True, budget_retest=False) is False
    assert conditioning_passes(3.2, 3.1, complete=True, budget_retest=False) is False


@pytest.mark.parametrize(
    ("complete", "budget_retest"), [(False, False), (True, True)]
)
def test_conditioning_check_is_pending_on_incomplete_or_retest(
    complete: bool, budget_retest: bool
) -> None:
    assert (
        conditioning_passes(
            3.0, 3.1, complete=complete, budget_retest=budget_retest
        )
        is None
    )
