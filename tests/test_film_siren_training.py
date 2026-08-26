from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mcar.data import Normalization
from mcar.paths import project_root
from mcar.training.train_film_siren import (
    conditioned_coordinate_block,
    read_common_grid,
    split_subject_paths,
)


ROOT = project_root()
DATASET_ROOT = ROOT / "data" / "processed" / "sonicom_residual_q26_v1"
SPLIT_CSV = ROOT / "configs" / "data" / "sonicom_subject_split_v1.csv"


def test_stage_b_uses_complete_frozen_train_and_validation_splits() -> None:
    train = split_subject_paths(DATASET_ROOT, SPLIT_CSV, "train")
    validation = split_subject_paths(DATASET_ROOT, SPLIT_CSV, "val")
    assert len(train) == 262
    assert len(validation) == 44
    assert {item[0] for item in train}.isdisjoint(
        {item[0] for item in validation}
    )
    with pytest.raises(PermissionError, match="never permits test"):
        split_subject_paths(DATASET_ROOT, SPLIT_CSV, "test")


def test_interpolation_query_grid_excludes_all_q26_directions() -> None:
    train = split_subject_paths(DATASET_ROOT, SPLIT_CSV, "train")
    directions, frequency, interpolation_mask, weights = read_common_grid(train[0][2])
    assert directions.shape == (793, 6)
    assert frequency.shape == (463,)
    assert np.count_nonzero(interpolation_mask) == 767
    assert weights.shape == (793,)
    assert np.sum(weights[interpolation_mask]) == pytest.approx(0.96206826)


def gate_normalization() -> Normalization:
    return Normalization(
        mca_mean=10.0,
        mca_std=2.0,
        correction_mean=0.0,
        correction_std=1.0,
        target_mean=0.0,
        target_std=1.0,
        log_frequency_mean=0.0,
        log_frequency_std=1.0,
    )


def test_gate_global_control_appends_zeros_without_local_mca() -> None:
    xyz = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    frequency = torch.tensor([[-1.0, -0.5], [1.0, 0.5]])
    query = conditioned_coordinate_block(
        xyz,
        frequency,
        "global_zero_local",
        None,
        gate_normalization(),
    )
    assert query.shape == (4, 7)
    torch.testing.assert_close(query[:, -2:], torch.zeros(4, 2))
    with pytest.raises(ValueError, match="must not read"):
        conditioned_coordinate_block(
            xyz,
            frequency,
            "global_zero_local",
            np.zeros((2, 2, 2), dtype=np.float32),
            gate_normalization(),
        )


def test_gate_local_mca_is_normalized_and_aligned_with_queries() -> None:
    xyz = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    frequency = torch.tensor([[-1.0, -0.5], [1.0, 0.5]])
    local_mca = np.asarray(
        [
            [[10.0, 12.0], [14.0, 16.0]],
            [[8.0, 6.0], [4.0, 2.0]],
        ],
        dtype=np.float32,
    )
    query = conditioned_coordinate_block(
        xyz,
        frequency,
        "global_plus_local_mca",
        local_mca,
        gate_normalization(),
    )
    expected = torch.tensor(
        [[0.0, -1.0], [1.0, -2.0], [2.0, -3.0], [3.0, -4.0]]
    )
    torch.testing.assert_close(query[:, -2:], expected)
    with pytest.raises(ValueError, match="requires local MCA"):
        conditioned_coordinate_block(
            xyz,
            frequency,
            "global_plus_local_mca",
            None,
            gate_normalization(),
        )


@pytest.mark.parametrize(
    ("name", "scope"),
    [
        ("global_zero_local", "global_zero_local"),
        ("global_plus_local_mca", "global_plus_local_mca"),
    ],
)
def test_global_local_gate_configs_match_protocol(name: str, scope: str) -> None:
    path = (
        ROOT
        / "configs"
        / "experiments"
        / f"sonicom_film_siren_b_gate_{name}_seed20260821_e150.json"
    )
    configuration = json.loads(path.read_text(encoding="utf-8"))
    assert configuration["conditioning_scope"] == scope
    assert configuration["model"]["coordinate_dimension"] == 7
    assert configuration["model"]["latent_dimension"] == 128
    assert configuration["model"]["modulation_variant"] == "full"
    assert configuration["model"]["placement"] == "all"
    assert configuration["cycles"] == 150
    assert configuration["seed"] == 20260821
    assert configuration["require_clean_git"] is True


@pytest.mark.parametrize("latent_dimension", [64, 128, 256])
def test_latent_screening_configs_match_protocol(latent_dimension: int) -> None:
    path = (
        ROOT
        / "configs"
        / "experiments"
        / f"sonicom_film_siren_b_latent_{latent_dimension}.json"
    )
    configuration = json.loads(path.read_text(encoding="utf-8"))
    assert configuration["experiment_type"] == "joint_film_siren_stage_b"
    assert configuration["condition_encoder"]["latent_dimension"] == latent_dimension
    assert configuration["model"]["latent_dimension"] == latent_dimension
    assert configuration["model"]["modulation_variant"] == "phase"
    assert configuration["model"]["placement"] == "hidden"
    assert configuration["cycles"] == 100
    assert configuration["steps_per_cycle"] == 262
    assert configuration["seed"] == 20260821
    assert configuration["require_clean_git"] is True
