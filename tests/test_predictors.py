from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from mcar.models.siren import Siren, SirenConfig
from mcar.predictors import ResidualPredictor, SirenPredictor
from mcar.training.train_siren import frequency_coordinates


SCRATCH = Path("artifacts") / "_predictor_test_tmp"


@pytest.fixture()
def siren_fixture() -> tuple[Path, Path, Siren]:
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    source = SCRATCH / "q26.h5"
    frequency = np.linspace(100.0, 1000.0, 9, dtype=np.float32)
    directions = np.zeros((7, 6), dtype=np.float32)
    directions[:, 2:5] = np.random.default_rng(4).normal(size=(7, 3))
    with h5py.File(source, "w") as handle:
        handle.create_dataset("direction_features", data=directions)
        handle.create_dataset("frequency_hz", data=frequency.reshape(-1, 1))
    configuration = SirenConfig(
        input_dimension=5,
        hidden_width=12,
        sine_layer_count=3,
        first_omega=20.0,
        hidden_omega=20.0,
        output_dimension=2,
    )
    torch.manual_seed(31)
    model = Siren(configuration).eval()
    checkpoint = SCRATCH / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "siren_configuration": configuration.to_dict(),
            "experiment_configuration": {
                "frequency_mapping": {
                    "mode": "dual",
                    "frequency_minimum_hz": 100.0,
                    "frequency_maximum_hz": 1000.0,
                }
            },
            "residual_db_conversion": {"target_mean": -0.25, "target_std": 4.0},
        },
        checkpoint,
    )
    try:
        yield source, checkpoint, model
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def test_siren_predictor_matches_direct_model(
    siren_fixture: tuple[Path, Path, Siren],
) -> None:
    source, checkpoint, model = siren_fixture
    predictor = SirenPredictor(
        checkpoint,
        device=torch.device("cpu"),
        directions_per_block=3,
    )
    actual = predictor.predict_residual_db(source)
    with h5py.File(source, "r") as handle:
        directions = np.asarray(handle["direction_features"][:], dtype=np.float32)
        frequency = np.asarray(handle["frequency_hz"][:]).reshape(-1)
    frequency_coordinate = frequency_coordinates(
        frequency,
        "dual",
        100.0,
        1000.0,
    )
    query = np.concatenate(
        (
            np.repeat(directions[:, None, 2:5], frequency.size, axis=1),
            np.repeat(frequency_coordinate[None, :, :], directions.shape[0], axis=0),
        ),
        axis=-1,
    ).reshape(-1, 5)
    with torch.no_grad():
        expected = (
            model(torch.from_numpy(query)).reshape(7, 9, 2).permute(2, 0, 1)
            * 4.0
            - 0.25
        ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert isinstance(predictor, ResidualPredictor)
