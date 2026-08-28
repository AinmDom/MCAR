from __future__ import annotations

import json

import torch

from mcar.models.film_siren import (
    ConditionEncoderConfig,
    FilmSiren,
    FilmSirenConfig,
)
from mcar.models.film_siren_spectral_cnn import (
    FilmSirenSpectralCNN,
    SpectralRefinerConfig,
)
from mcar.paths import project_root
from mcar.training.train_film_siren import file_sha256


def make_model() -> FilmSirenSpectralCNN:
    backbone = FilmSiren(
        FilmSirenConfig(
            coordinate_dimension=7,
            hidden_width=16,
            sine_layer_count=3,
            first_omega=20.0,
            hidden_omega=20.0,
            output_dimension=2,
            latent_dimension=8,
            modulation_variant="full",
            placement="all",
            modulator_width=12,
        ),
        ConditionEncoderConfig(
            frequency_count=31,
            convolution_channels=4,
            direction_embedding_dimension=12,
            latent_dimension=8,
        ),
    )
    return FilmSirenSpectralCNN(
        backbone,
        SpectralRefinerConfig(
            channels=8,
            dilation_schedule=(1, 2),
            kernel_size=5,
            condition_width=6,
        ),
        freeze_film_siren=True,
    )


def make_grid() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(71)
    direction_count = 5
    frequency_count = 31
    coordinates = torch.randn(
        direction_count * frequency_count,
        7,
        generator=generator,
    )
    latent = torch.randn(1, 8, generator=generator)
    normalized_mca = torch.randn(
        2, direction_count, frequency_count, generator=generator
    )
    normalized_correction = torch.randn(
        2, direction_count, frequency_count, generator=generator
    )
    normalized_log_frequency = torch.linspace(-1.5, 1.5, frequency_count)
    direction_xyz = torch.randn(direction_count, 3, generator=generator)
    direction_xyz = direction_xyz / torch.linalg.vector_norm(
        direction_xyz, dim=1, keepdim=True
    )
    return (
        coordinates,
        latent,
        normalized_mca,
        normalized_correction,
        normalized_log_frequency,
        direction_xyz,
    )


def test_zero_initialized_refiner_is_exactly_the_film_siren() -> None:
    model = make_model().eval()
    grid = make_grid()
    with torch.no_grad():
        final, base, delta = model.forward_grid(*grid)
        expected = (
            model.film_siren(grid[0], grid[1])
            .reshape(5, 31, 2)
            .permute(2, 0, 1)
        )
    torch.testing.assert_close(base, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0.0, atol=0.0)
    torch.testing.assert_close(final, expected, rtol=0.0, atol=0.0)


def test_cnn_only_backward_keeps_film_siren_frozen() -> None:
    model = make_model().train()
    assert model.film_siren.training is False
    final, _, _ = model.forward_grid(*make_grid())
    final.square().mean().backward()
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.film_siren.parameters()
    )
    assert model.spectral_cnn.output.weight.grad is not None
    assert bool(torch.all(torch.isfinite(model.spectral_cnn.output.weight.grad)))
    assert float(torch.linalg.vector_norm(model.spectral_cnn.output.weight.grad)) > 0.0


def test_refiner_configuration_round_trip_is_json_compatible() -> None:
    configuration = make_model().refiner_configuration_dict()
    assert configuration == {
        "input_channels": 7,
        "channels": 8,
        "dilation_schedule": [1, 2],
        "kernel_size": 5,
        "condition_width": 6,
    }


def test_d1_configuration_freezes_the_parent_and_mcar_interface() -> None:
    root = project_root()
    path = (
        root
        / "configs"
        / "experiments"
        / "sonicom_film_siren_spectral_cnn_d1_seed20260821_e40.json"
    )
    configuration = json.loads(path.read_text(encoding="utf-8"))
    assert configuration["experiment_type"] == "film_siren_spectral_cnn_stage_d"
    assert configuration["conditioning_scope"] == "global_plus_local_mca"
    assert configuration["freeze_film_siren"] is True
    assert configuration["spectral_refiner"] == {
        "input_channels": 7,
        "channels": 48,
        "dilation_schedule": [1, 2, 4, 8],
        "kernel_size": 7,
        "condition_width": 64,
    }
    assert configuration["cycles"] == 40
    assert configuration["checkpoint_policy"] == "validation_best"
    checkpoint = root / configuration["initial_film_checkpoint"]
    assert file_sha256(checkpoint).upper() == configuration[
        "initial_film_checkpoint_sha256"
    ]
