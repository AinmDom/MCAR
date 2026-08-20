from __future__ import annotations

import json

import numpy as np
import pytest

from mcar.paths import project_root
from mcar.training.train_film_siren import (
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
