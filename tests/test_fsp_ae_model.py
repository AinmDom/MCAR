"""Offline shape, gradient, and parameter-count checks for FSP-AE."""

from __future__ import annotations

import torch

from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder, parameter_count


def main() -> None:
    torch.manual_seed(20260806)
    model = FreqSrcPosCondAutoEncoder()
    model.set_stats(-6.0, 11.0, "synthetic", "hrtf_mag")
    model.set_stats(0.0, 3.7e-4, "synthetic", "itd")
    magnitude = torch.randn(1, 4, 2, 8) * 8.0 - 5.0
    itd = torch.randn(1, 4) * 3.0e-4
    frequency = torch.linspace(125.0, 16_000.0, 8).unsqueeze(0)
    measurement_positions = torch.randn(1, 4, 3)
    measurement_positions = (
        1.5
        * measurement_positions
        / torch.linalg.vector_norm(measurement_positions, dim=-1, keepdim=True)
    )
    target_positions = torch.randn(1, 6, 3)
    target_positions = (
        1.5
        * target_positions
        / torch.linalg.vector_norm(target_positions, dim=-1, keepdim=True)
    )

    predicted_magnitude, predicted_itd = model(
        magnitude,
        itd,
        frequency,
        measurement_positions,
        target_positions,
        "synthetic",
    )
    assert predicted_magnitude.shape == (1, 6, 2, 8)
    assert predicted_itd.shape == (1, 6)
    loss = predicted_magnitude.square().mean() + predicted_itd.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )
    assert parameter_count(model) == 235_065
    print(
        {
            "status": "passed",
            "parameter_count": parameter_count(model),
            "magnitude_shape": list(predicted_magnitude.shape),
            "itd_shape": list(predicted_itd.shape),
        }
    )


if __name__ == "__main__":
    main()
