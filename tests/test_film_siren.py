from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mcar.models.film_siren import (
    ConditionEncoderConfig,
    FilmSiren,
    FilmSirenConfig,
    Q26ConditionEncoder,
    placement_layer_indices,
)
from mcar.models.siren import Siren, SirenConfig


def small_encoder_config(latent_dimension: int = 8) -> ConditionEncoderConfig:
    return ConditionEncoderConfig(
        frequency_count=31,
        convolution_channels=4,
        direction_embedding_dimension=12,
        latent_dimension=latent_dimension,
    )


def small_model_config(
    variant: str = "phase",
    placement: str = "hidden",
) -> FilmSirenConfig:
    return FilmSirenConfig(
        coordinate_dimension=5,
        hidden_width=16,
        sine_layer_count=6,
        first_omega=20.0,
        hidden_omega=20.0,
        latent_dimension=8,
        modulation_variant=variant,
        placement=placement,
        modulator_width=12,
    )


def make_condition() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    magnitude = torch.randn(2, 2, 7, 31, generator=generator)
    xyz = torch.randn(2, 7, 3, generator=generator)
    xyz = xyz / torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    mask = torch.ones(2, 7, dtype=torch.bool)
    return magnitude, xyz, mask


def test_encoder_shape_finite_and_backward() -> None:
    encoder = Q26ConditionEncoder(small_encoder_config())
    magnitude, xyz, mask = make_condition()
    latent = encoder(magnitude, xyz, mask)
    assert latent.shape == (2, 8)
    assert bool(torch.all(torch.isfinite(latent)).item())
    latent.square().mean().backward()
    assert all(
        parameter.grad is not None and bool(torch.all(torch.isfinite(parameter.grad)))
        for parameter in encoder.parameters()
    )


def test_encoder_is_direction_permutation_invariant() -> None:
    encoder = Q26ConditionEncoder(small_encoder_config()).eval()
    magnitude, xyz, mask = make_condition()
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    with torch.no_grad():
        base = encoder(magnitude, xyz, mask)
        permuted = encoder(
            magnitude[:, :, permutation, :],
            xyz[:, permutation, :],
            mask[:, permutation],
        )
    torch.testing.assert_close(permuted, base, rtol=1e-6, atol=1e-6)


def test_encoder_ignores_masked_padding_values() -> None:
    encoder = Q26ConditionEncoder(small_encoder_config()).eval()
    magnitude, xyz, mask = make_condition()
    mask[:, -2:] = False
    changed_magnitude = magnitude.clone()
    changed_xyz = xyz.clone()
    changed_magnitude[:, :, -2:, :] = 1000.0
    changed_xyz[:, -2:, :] = -1000.0
    with torch.no_grad():
        base = encoder(magnitude, xyz, mask)
        changed = encoder(changed_magnitude, changed_xyz, mask)
    torch.testing.assert_close(changed, base, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("variant", ["amplitude", "phase", "full"])
def test_zero_initialized_film_matches_plain_siren(variant: str) -> None:
    plain_configuration = SirenConfig(
        input_dimension=5,
        hidden_width=16,
        sine_layer_count=6,
        first_omega=20.0,
        hidden_omega=20.0,
        output_dimension=2,
    )
    torch.manual_seed(23)
    plain = Siren(plain_configuration)
    torch.manual_seed(23)
    film = FilmSiren(
        small_model_config(variant=variant),
        small_encoder_config(),
    )
    coordinates = torch.randn(2, 19, 5)
    latent = torch.randn(2, 8)
    with torch.no_grad():
        expected = torch.stack([plain(item) for item in coordinates])
        actual = film(coordinates, latent)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("variant", ["amplitude", "phase", "full"])
def test_film_variants_forward_and_gradients(variant: str) -> None:
    model = FilmSiren(
        small_model_config(variant=variant),
        small_encoder_config(),
    )
    magnitude, xyz, mask = make_condition()
    coordinates = torch.randn(2, 13, 5)
    prediction = model.forward_from_condition(coordinates, magnitude, xyz, mask)
    assert prediction.shape == (2, 13, 2)
    prediction.square().mean().backward()
    assert bool(torch.all(torch.isfinite(prediction)).item())


def test_concat_variant_has_explicit_no_placement_branch() -> None:
    configuration = small_model_config(variant="concat", placement="none")
    model = FilmSiren(configuration, small_encoder_config())
    magnitude, xyz, mask = make_condition()
    coordinates = torch.randn(2, 11, 5)
    assert model.forward_from_condition(
        coordinates,
        magnitude,
        xyz,
        mask,
    ).shape == (2, 11, 2)
    with pytest.raises(ValueError, match="placement='none'"):
        replace(configuration, placement="all").validate()


def test_placement_definitions_are_exact() -> None:
    base = small_model_config()
    assert placement_layer_indices(replace(base, placement="late")) == (3, 4, 5)
    assert placement_layer_indices(replace(base, placement="hidden")) == (
        1,
        2,
        3,
        4,
        5,
    )
    assert placement_layer_indices(replace(base, placement="all")) == tuple(range(6))


def test_all_masked_encoder_input_is_rejected() -> None:
    encoder = Q26ConditionEncoder(small_encoder_config())
    magnitude, xyz, mask = make_condition()
    mask[0] = False
    with pytest.raises(ValueError, match="valid direction"):
        encoder(magnitude, xyz, mask)


def test_zero_initialized_modulation_statistics_report_identity() -> None:
    model = FilmSiren(small_model_config(variant="full"), small_encoder_config())
    statistics = model.modulation_statistics(torch.randn(3, 8))
    assert statistics["variant"] == "full"
    assert set(statistics["layers"]) == {"2", "3", "4", "5", "6"}
    for layer in statistics["layers"].values():
        assert layer["gamma"]["mean"] == pytest.approx(1.0)
        assert layer["beta"]["mean"] == pytest.approx(0.0)
        assert layer["amplitude"]["mean"] == pytest.approx(1.0)
        assert layer["gamma"]["saturation_fraction"] == pytest.approx(0.0)


def test_film_siren_state_dict_round_trip() -> None:
    configuration = small_model_config(variant="full")
    encoder_configuration = small_encoder_config()
    torch.manual_seed(41)
    source = FilmSiren(configuration, encoder_configuration).eval()
    target = FilmSiren(configuration, encoder_configuration).eval()
    target.load_state_dict(source.state_dict())
    magnitude, xyz, mask = make_condition()
    coordinates = torch.randn(2, 9, 5)
    with torch.no_grad():
        expected = source.forward_from_condition(
            coordinates,
            magnitude,
            xyz,
            mask,
        )
        actual = target.forward_from_condition(
            coordinates,
            magnitude,
            xyz,
            mask,
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
