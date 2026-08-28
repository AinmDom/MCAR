from __future__ import annotations

import torch

from mcar.models.bounded_mcar_film_correction import (
    BoundedCorrectionConfig,
    BoundedMcarFilmCorrection,
)
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.models.residual_mlp_cnn import ResidualMLPCNN


def make_model(maximum_mix: float = 0.5) -> BoundedMcarFilmCorrection:
    film = FilmSiren(
        FilmSirenConfig(
            coordinate_dimension=7,
            hidden_width=16,
            sine_layer_count=2,
            output_dimension=2,
            latent_dimension=8,
            modulator_width=12,
        ),
        ConditionEncoderConfig(
            frequency_count=9,
            convolution_channels=4,
            direction_embedding_dimension=6,
            latent_dimension=8,
        ),
    )
    return BoundedMcarFilmCorrection(
        film,
        ResidualMLPCNN(mlp_width=8, mlp_block_count=1, cnn_channels=4),
        ResidualMLPCNN(mlp_width=8, mlp_block_count=1, cnn_channels=4),
        BoundedCorrectionConfig(maximum_mix=maximum_mix),
    )


def test_zero_initialization_is_exact_mcar_identity() -> None:
    torch.manual_seed(7)
    model = make_model()
    coordinates = torch.randn(3 * 64, 7)
    latent = torch.randn(1, 8)
    point_features = torch.randn(3, 2, 64, 7)
    final, base, correction, gate = model.forward_grid(
        coordinates, latent, point_features
    )
    torch.testing.assert_close(final, base, rtol=0.0, atol=0.0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gate, torch.zeros_like(gate), rtol=0.0, atol=0.0)


def test_only_gate_is_trainable_and_bound_is_respected() -> None:
    model = make_model(maximum_mix=0.25)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable == ["gate_output.weight", "gate_output.bias"]
    with torch.no_grad():
        model.gate_output.bias.fill_(100.0)
    final, base, correction, gate = model.forward_grid(
        torch.randn(2 * 64, 7),
        torch.randn(1, 8),
        torch.randn(2, 2, 64, 7),
    )
    assert float(gate.detach().abs().max()) <= 0.25
    torch.testing.assert_close(final, base + correction)


def test_backbones_stay_in_evaluation_mode() -> None:
    model = make_model()
    model.train()
    assert not model.film_siren.training
    assert not model.previous_mcar.training
    assert not model.candidate_mcar.training
    assert model.gate_output.training
