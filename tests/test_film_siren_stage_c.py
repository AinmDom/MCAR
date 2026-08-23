from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mcar.paths import project_root
from mcar.training.film_siren_stage_c import (
    StageCLossConfiguration,
    calculate_stage_c_losses,
)
from mcar.training.train_film_siren import read_common_grid, split_subject_paths
from mcar.training.train_film_siren_stage_c import (
    horizontal_interpolation_indices,
    learning_rate_for_cycle,
)
from mcar.training.train_mlp_v2 import (
    make_erb_center_frequencies_hz,
    make_erb_weights,
)


ROOT = project_root()
DATASET_ROOT = ROOT / "data" / "processed" / "sonicom_residual_q26_v1"
SPLIT_CSV = ROOT / "configs" / "data" / "sonicom_subject_split_v1.csv"
CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "sonicom_film_siren_c1_lr1e4_adam_seed20260821_e150.json"
)


def test_stage_c_initial_config_matches_preregistered_protocol() -> None:
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert configuration["experiment_type"] == "joint_film_siren_stage_c"
    assert configuration["model"]["latent_dimension"] == 128
    assert configuration["model"]["modulation_variant"] == "full"
    assert configuration["model"]["placement"] == "all"
    assert configuration["optimizer"] == {
        "name": "Adam",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
    }
    assert configuration["scheduler"]["name"] == "constant"
    assert configuration["scheduler"]["horizon_cycles"] == 150
    assert configuration["cycles"] == 150
    assert configuration["steps_per_cycle"] == 262
    assert configuration["global_directions_per_step"] == 16
    assert configuration["horizontal_directions_per_step"] == 16
    objective = configuration["objective"]
    assert objective["erb_weight"] == 0.75
    assert objective["high_frequency_weight"] == 0.25
    assert objective["strict_ild_weight"] == 0.75
    assert objective["spectral_band_ild_weight"] == 0.05
    assert objective["high_frequency_first_difference_weight"] == 0.0
    assert objective["notch_depth_weight"] == 0.0
    assert configuration["require_clean_git"] is True


@pytest.mark.parametrize(
    ("suffix", "learning_rate"),
    [("lr3e5", 3e-5), ("lr1e4", 1e-4), ("lr3e4", 3e-4)],
)
def test_c1_learning_rate_configs_change_only_preregistered_identity_fields(
    suffix: str, learning_rate: float
) -> None:
    path = (
        ROOT
        / "configs"
        / "experiments"
        / f"sonicom_film_siren_c1_{suffix}_adam_seed20260821_e150.json"
    )
    candidate = json.loads(path.read_text(encoding="utf-8"))
    baseline = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert candidate["optimizer"]["learning_rate"] == learning_rate
    for configuration in (candidate, baseline):
        configuration["optimizer"]["learning_rate"] = None
        for field in (
            "created_on",
            "experiment_id",
            "model_version",
            "run_name",
        ):
            configuration[field] = None
    assert candidate == baseline


def test_stage_c_horizontal_support_is_frozen_and_test_is_forbidden() -> None:
    train = split_subject_paths(DATASET_ROOT, SPLIT_CSV, "train")
    directions, _, interpolation_mask, _ = read_common_grid(train[0][2])
    indices = horizontal_interpolation_indices(directions, interpolation_mask)
    assert indices.size == 72
    assert np.all(interpolation_mask[indices])
    assert np.all(np.abs(directions[indices, 1]) <= 1e-6)
    with pytest.raises(PermissionError, match="never permits test"):
        split_subject_paths(DATASET_ROOT, SPLIT_CSV, "test")


def test_scheduler_trajectory_keeps_horizon_separate_from_stop_cycle() -> None:
    base = 1e-4
    assert learning_rate_for_cycle(base, "constant", 20, 150, 0) == base
    assert learning_rate_for_cycle(base, "cosine", 1, 150, 0) == base
    assert learning_rate_for_cycle(base, "cosine", 75, 150, 0) > 0.0
    assert learning_rate_for_cycle(base, "warmup_cosine", 1, 150, 5) == pytest.approx(
        2e-5
    )
    assert learning_rate_for_cycle(base, "warmup_cosine", 5, 150, 5) == base
    assert learning_rate_for_cycle(base, "warmup_cosine", 150, 150, 5) < 1e-7


def test_stage_c_dual_sampling_loss_has_finite_gradient_and_exact_increment() -> None:
    torch.manual_seed(7)
    global_directions = 5
    horizontal_directions = 4
    frequency_count = 65
    frequency = np.linspace(100.0, 20_000.0, frequency_count, dtype=np.float32)
    weights = torch.from_numpy(make_erb_weights(frequency))
    log_weights = torch.log(torch.clamp(weights, min=1e-12)).view(
        1, 1, weights.shape[0], weights.shape[1]
    )
    centers = torch.from_numpy(make_erb_center_frequencies_hz())
    global_prediction = torch.randn(
        2, global_directions, frequency_count, requires_grad=True
    )
    horizontal_prediction = torch.randn(
        2, horizontal_directions, frequency_count, requires_grad=True
    )
    global_target = torch.randn(2, global_directions, frequency_count)
    horizontal_target = torch.randn(2, horizontal_directions, frequency_count)
    global_features = torch.zeros(global_directions, 6)
    global_features[:, 3] = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0])
    global_features[:, 5] = torch.linspace(1.0, 2.0, global_directions)
    horizontal_features = torch.zeros(horizontal_directions, 6)
    horizontal_features[:, 5] = torch.linspace(1.0, 2.0, horizontal_directions)
    strict = {
        "mca_selected_phase_rad": torch.zeros(
            2, horizontal_directions, frequency_count
        ),
        "mca_outside_real": torch.empty(2, horizontal_directions, 0),
        "mca_outside_imag": torch.empty(2, horizontal_directions, 0),
        "selected_bin_indices_zero_based": torch.arange(frequency_count),
        "outside_bin_indices_zero_based": torch.empty(0, dtype=torch.int64),
        "reference_ild_db": torch.zeros(horizontal_directions),
        "single_sided_frequency_count": frequency_count,
        "hrir_length": 128,
    }
    baseline_configuration = StageCLossConfiguration(
        strict_ild_weight=0.0, spectral_band_ild_weight=0.0
    )
    stage_c_configuration = StageCLossConfiguration()
    arguments = (
        global_prediction,
        global_target,
        global_target,
        torch.zeros_like(global_target),
        global_features,
        horizontal_prediction,
        horizontal_target,
        torch.zeros_like(horizontal_target),
        horizontal_features,
        strict,
        torch.from_numpy(frequency),
        log_weights,
        centers,
        0.0,
        1.0,
    )
    baseline, _ = calculate_stage_c_losses(*arguments, baseline_configuration)
    total, metrics = calculate_stage_c_losses(*arguments, stage_c_configuration)
    expected = (
        stage_c_configuration.strict_ild_weight * metrics.ild_mae_db
        + stage_c_configuration.spectral_band_ild_weight
        * metrics.spectral_band_ild_smooth_l1_db
    )
    torch.testing.assert_close(total - baseline, torch.tensor(expected), rtol=1e-5, atol=1e-6)
    assert torch.isfinite(total)
    total.backward()
    assert global_prediction.grad is not None
    assert horizontal_prediction.grad is not None
    assert torch.all(torch.isfinite(global_prediction.grad))
    assert torch.all(torch.isfinite(horizontal_prediction.grad))
