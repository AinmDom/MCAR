"""Verify memory-bounded decoder chunks preserve the full FSP-AE gradient."""

from __future__ import annotations

import copy

import torch

from mcar.fsp_ae_signal import lsd_loss
from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder


def configure(model: FreqSrcPosCondAutoEncoder) -> None:
    model.set_stats(-20.0, 15.0, "synthetic", "hrtf_mag")
    model.set_stats(0.0, 4.0e-4, "synthetic", "itd")


def main() -> None:
    torch.manual_seed(20260806)
    direct = FreqSrcPosCondAutoEncoder()
    configure(direct)
    chunked = copy.deepcopy(direct)

    measurement_magnitude = torch.randn(1, 4, 2, 8) * 12.0 - 20.0
    measurement_itd = torch.randn(1, 4) * 3.0e-4
    frequency = torch.linspace(125.0, 16_000.0, 8).unsqueeze(0)
    measurement_positions = torch.randn(1, 4, 3)
    target_positions = torch.randn(1, 6, 3)
    target_magnitude = torch.randn(1, 6, 2, 8) * 12.0 - 20.0
    target_itd = torch.randn(1, 6) * 3.0e-4
    inputs = (
        measurement_magnitude,
        measurement_itd,
        frequency,
        measurement_positions,
    )

    predicted_magnitude, predicted_itd = direct(
        *inputs, target_positions, "synthetic"
    )
    full_loss = lsd_loss(predicted_magnitude, target_magnitude) + 2500.0 * (
        predicted_itd - target_itd
    ).abs().mean()
    full_loss.backward()

    prototype = chunked.encode(*inputs, dataset_name="synthetic")
    prototype_leaf = prototype.detach().requires_grad_(True)
    for start in range(0, target_positions.shape[1], 2):
        stop = start + 2
        chunk_magnitude, chunk_itd = chunked.decode(
            prototype_leaf,
            frequency,
            target_positions[:, start:stop],
            dataset_name="synthetic",
        )
        chunk_loss = lsd_loss(
            chunk_magnitude, target_magnitude[:, start:stop]
        ) + 2500.0 * (
            chunk_itd - target_itd[:, start:stop]
        ).abs().mean()
        (chunk_loss * (stop - start) / target_positions.shape[1]).backward()
    assert prototype_leaf.grad is not None
    prototype.backward(prototype_leaf.grad)

    maximum_error = 0.0
    for direct_parameter, chunked_parameter in zip(
        direct.parameters(), chunked.parameters()
    ):
        assert direct_parameter.grad is not None
        assert chunked_parameter.grad is not None
        maximum_error = max(
            maximum_error,
            float(
                torch.max(
                    torch.abs(
                        direct_parameter.grad - chunked_parameter.grad
                    )
                )
            ),
        )
    if maximum_error > 2e-5:
        raise AssertionError(f"chunked gradient error {maximum_error}")
    print({"status": "passed", "gradient_max_abs_error": maximum_error})


if __name__ == "__main__":
    main()
