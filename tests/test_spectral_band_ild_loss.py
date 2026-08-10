"""CPU regression tests for the v3.4b spectral-band ILD objective."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mcar.losses import spectral_band_ild_smooth_l1_and_mae
from mcar.training.train_mlp_cnn_v3 import training_stage


def main() -> None:
    reference = torch.zeros(2, 2, 5)
    direction_weights = torch.tensor([0.75, 0.25])
    raw_band_weights = torch.tensor(
        [
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0],
        ]
    )
    raw_band_weights = raw_band_weights / raw_band_weights.sum(
        dim=1, keepdim=True
    )
    log_band_weights = torch.log(
        torch.clamp(raw_band_weights, min=1e-12)
    ).view(1, 1, 3, 5)
    band_centers_hz = torch.tensor([100.0, 1000.0, 19000.0])

    zero_loss, zero_mae = spectral_band_ild_smooth_l1_and_mae(
        reference,
        reference,
        direction_weights,
        log_band_weights,
        band_centers_hz,
        minimum_band_center_hz=200.0,
        maximum_band_center_hz=18000.0,
        beta_db=0.5,
    )
    torch.testing.assert_close(zero_loss, torch.tensor(0.0))
    torch.testing.assert_close(zero_mae, torch.tensor(0.0))

    corrected = reference.clone()
    corrected[0, 0, :] += 1.0
    corrected[0, 1, :] += 3.0
    corrected.requires_grad_(True)
    smooth_l1, mae = spectral_band_ild_smooth_l1_and_mae(
        corrected,
        reference,
        direction_weights,
        log_band_weights,
        band_centers_hz,
        minimum_band_center_hz=200.0,
        maximum_band_center_hz=18000.0,
        beta_db=0.5,
    )
    # SmoothL1(beta=.5): |1| -> .75 and |3| -> 2.75.
    torch.testing.assert_close(smooth_l1, torch.tensor(1.25), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(mae, torch.tensor(1.5), atol=1e-6, rtol=0.0)
    smooth_l1.backward()
    if corrected.grad is None or not bool(
        torch.all(torch.isfinite(corrected.grad)).item()
    ):
        raise AssertionError("Spectral-band ILD gradients are invalid")
    if not bool(torch.any(corrected.grad != 0.0).item()):
        raise AssertionError("Spectral-band ILD gradient is unexpectedly zero")

    arguments = SimpleNamespace(
        high_frequency_first_difference_weight=0.0,
        high_frequency_second_difference_weight=0.0,
        spectral_band_ild_weight=0.10,
        unfreeze_mlp=False,
        dual_sampling_strict_ild=True,
        global_context_attention=False,
        ild_loss_mode="strict_hrir",
        horizontal_only=False,
    )
    assert training_stage(arguments).endswith("spectral_band_ild_v34b")
    print(
        {
            "status": "passed",
            "spectral_band_ild_smooth_l1_db": float(smooth_l1.detach()),
            "spectral_band_ild_mae_db": float(mae.detach()),
            "training_stage": training_stage(arguments),
        }
    )


if __name__ == "__main__":
    main()
