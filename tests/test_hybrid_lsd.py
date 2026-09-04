"""LSD reduction parity, zero-error stability and matched experimental controls."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mcar.evaluation.secondary_metrics import full_sphere_lsd
from mcar.training.film_siren_stage_c import StageCLossConfiguration, spectral_lsd_db


def test_lsd_matches_numpy_evaluator_and_is_not_global_rmse():
    error = np.random.default_rng(42).normal(size=(2, 7, 11))
    error *= np.arange(1, 8)[None, :, None]
    weights = np.arange(1, 8, dtype=float)
    expected, _, _ = full_sphere_lsd(error, np.zeros_like(error), np.ones(7, bool), weights)
    result = spectral_lsd_db(torch.tensor(error), torch.tensor(weights))
    assert result.item() == pytest.approx(expected, abs=1e-12)
    rmse = np.sqrt(np.sum(np.mean(error ** 2, axis=(0, 2)) * weights / weights.sum()))
    assert abs(result.item() - rmse) > 0.1


def test_lsd_zero_error_has_zero_finite_gradient():
    error = torch.zeros(2, 3, 463, requires_grad=True)
    loss = spectral_lsd_db(error, torch.ones(3), 1e-6)
    loss.backward()
    assert loss.item() == pytest.approx(1e-6)
    assert torch.equal(error.grad, torch.zeros_like(error))
    assert spectral_lsd_db(error.detach(), torch.ones(3)).item() == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_lsd_weights(value):
    with pytest.raises(ValueError):
        StageCLossConfiguration(lsd_weight=value).validate()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_lsd_epsilon(value):
    with pytest.raises(ValueError):
        StageCLossConfiguration(lsd_epsilon_db=value).validate()


@pytest.mark.parametrize("seed", [20260821, 20260822, 20260823])
def test_b_configuration_only_changes_identity_and_lsd(seed):
    root = Path(__file__).resolve().parents[1] / "configs/experiments"
    old = json.loads((root / f"sonicom_film_siren_spectral_cnn_final_seed{seed}_e190.json").read_text())
    new = json.loads((root / f"sonicom_film_siren_spectral_cnn_lsd_b_seed{seed}_e190.json").read_text())
    for key in ("created_on", "experiment_id", "model_version", "protocol", "search_stage", "run_name"):
        old.pop(key)
        new.pop(key)
    assert new["objective"].pop("lsd_weight") == 1.0
    assert new["objective"].pop("lsd_epsilon_db") == 1e-6
    assert new == old
