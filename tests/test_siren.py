"""Unit and device smoke tests for the plain SIREN backbone."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from mcar.models.siren import (
    SineLayer,
    Siren,
    SirenConfig,
    reference_bias_bound,
    siren_weight_bound,
)
from mcar.training.train_siren import frequency_coordinates, verify_train_subject


def make_small_model() -> Siren:
    torch.manual_seed(7)
    return Siren(
        SirenConfig(
            input_dimension=4,
            hidden_width=32,
            sine_layer_count=3,
            first_omega=30.0,
            hidden_omega=20.0,
            output_dimension=2,
        )
    )


def assert_inside(values: torch.Tensor, bound: float) -> None:
    assert float(torch.max(values).item()) <= bound + 1e-7
    assert float(torch.min(values).item()) >= -bound - 1e-7


def test_forward_backward_and_shape_are_finite() -> None:
    model = make_small_model()
    coordinates = torch.randn(19, 4, requires_grad=True)
    output = model(coordinates)
    assert output.shape == (19, 2)
    assert bool(torch.all(torch.isfinite(output)).item())
    torch.square(output).mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.all(torch.isfinite(gradient)).item()) for gradient in gradients)


def test_initialization_bounds_are_explicit() -> None:
    model = make_small_model()
    first = model.sine_layers[0]
    assert isinstance(first, SineLayer)
    assert_inside(
        first.linear.weight,
        siren_weight_bound(4, 30.0, first=True),
    )
    assert_inside(first.linear.bias, reference_bias_bound(4))
    second = model.sine_layers[1]
    assert isinstance(second, SineLayer)
    assert_inside(
        second.linear.weight,
        siren_weight_bound(32, 20.0, first=False),
    )
    assert_inside(second.linear.bias, reference_bias_bound(32))
    assert_inside(
        model.output.weight,
        siren_weight_bound(32, 20.0, first=False),
    )
    assert_inside(model.output.bias, reference_bias_bound(32))


def test_checkpoint_round_trip_is_exact() -> None:
    model = make_small_model()
    coordinates = torch.randn(11, 4)
    expected = model(coordinates).detach()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "siren.pt"
        torch.save(model.state_dict(), path)
        restored = make_small_model()
        restored.load_state_dict(torch.load(path, weights_only=True))
    actual = restored(coordinates).detach()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_frequency_coordinate_definitions() -> None:
    frequencies = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)
    linear = frequency_coordinates(frequencies, "linear", 100.0, 10000.0)
    erb = frequency_coordinates(frequencies, "erb", 100.0, 10000.0)
    dual = frequency_coordinates(frequencies, "dual", 100.0, 10000.0)
    assert linear.shape == (3, 1)
    assert erb.shape == (3, 1)
    assert dual.shape == (3, 2)
    np.testing.assert_allclose(linear[[0, -1], 0], [-1.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(erb[[0, -1], 0], [-1.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(dual[:, 0], linear[:, 0], atol=0.0)
    np.testing.assert_allclose(dual[:, 1], erb[:, 0], atol=0.0)


def test_invalid_coordinate_dimension_fails() -> None:
    model = make_small_model()
    with pytest.raises(ValueError, match="coordinate dimension"):
        model(torch.randn(3, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fp32_and_amp_are_finite_and_close() -> None:
    model = make_small_model().cuda().eval()
    coordinates = torch.randn(128, 4, device="cuda")
    with torch.no_grad():
        fp32 = model(coordinates)
        with torch.amp.autocast("cuda", enabled=True):
            amp = model(coordinates)
    assert bool(torch.all(torch.isfinite(fp32)).item())
    assert bool(torch.all(torch.isfinite(amp)).item())
    torch.testing.assert_close(amp.float(), fp32, rtol=2e-2, atol=2e-2)


def test_reference_bias_bound_matches_linear_definition() -> None:
    assert math.isclose(reference_bias_bound(16), 0.25)


def test_single_subject_training_refuses_non_train_split() -> None:
    with TemporaryDirectory() as directory:
        split_csv = Path(directory) / "split.csv"
        with split_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("subject_id", "split"))
            writer.writeheader()
            writer.writerow({"subject_id": "P0001", "split": "val"})
        with pytest.raises(PermissionError, match="only permits train subjects"):
            verify_train_subject(split_csv, 1)
