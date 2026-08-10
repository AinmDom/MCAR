"""CPU regression tests for the v3.4a spectral-shape objective."""

from __future__ import annotations

import numpy as np
import torch

from mcar.losses import high_frequency_spectral_difference_mae
from mcar.training.train_mlp_v2 import calculate_losses, make_erb_weights


def difference_metrics(error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.zeros_like(error)
    direction_weights = torch.tensor([0.25, 0.75])
    frequency_hz = torch.tensor(
        [1000.0, 4000.0, 6000.0, 8000.0, 11000.0, 14000.0]
    )
    return high_frequency_spectral_difference_mae(
        error,
        reference,
        direction_weights,
        frequency_hz,
        minimum_frequency_hz=4000.0,
    )


def main() -> None:
    shape = (2, 2, 6)
    zeros = torch.zeros(shape)
    first, second = difference_metrics(zeros)
    assert first.item() == 0.0
    assert second.item() == 0.0

    constant = torch.full(shape, 3.0)
    first, second = difference_metrics(constant)
    assert first.item() == 0.0
    assert second.item() == 0.0

    bins = torch.arange(6, dtype=torch.float32).view(1, 1, -1)
    linear = bins.expand(shape)
    first, second = difference_metrics(linear)
    torch.testing.assert_close(first, torch.tensor(1.0))
    torch.testing.assert_close(second, torch.tensor(0.0))

    quadratic = bins.square().expand(shape).clone().requires_grad_(True)
    first, second = difference_metrics(quadratic)
    assert first.item() > 0.0
    torch.testing.assert_close(second, torch.tensor(2.0))
    (first + second).backward()
    assert quadratic.grad is not None
    assert bool(torch.all(torch.isfinite(quadratic.grad)).item())
    assert bool(torch.any(quadratic.grad != 0.0).item())

    frequency = np.array(
        [1000.0, 4000.0, 6000.0, 8000.0, 11000.0, 14000.0],
        dtype=np.float32,
    )
    log_erb_weights = torch.log(
        torch.clamp(
            torch.from_numpy(make_erb_weights(frequency)), min=1e-12
        )
    ).view(1, 1, -1, len(frequency))
    prediction_db = quadratic.detach()
    target_db = torch.zeros_like(prediction_db)
    mca_db = torch.zeros_like(prediction_db)
    directions = torch.zeros(2, 6)
    directions[:, 3] = torch.tensor([-1.0, 1.0])
    directions[:, 5] = torch.tensor([0.25, 0.75])
    old_total, old_metrics = calculate_losses(
        prediction_db,
        target_db,
        target_db,
        mca_db,
        directions,
        torch.from_numpy(frequency),
        log_erb_weights,
        target_mean=0.0,
        target_std=1.0,
        erb_weight=0.0,
        high_frequency_weight=0.0,
        ild_weight=0.0,
        direction_weighted_residual=True,
    )
    explicit_zero_total, _ = calculate_losses(
        prediction_db,
        target_db,
        target_db,
        mca_db,
        directions,
        torch.from_numpy(frequency),
        log_erb_weights,
        target_mean=0.0,
        target_std=1.0,
        erb_weight=0.0,
        high_frequency_weight=0.0,
        ild_weight=0.0,
        direction_weighted_residual=True,
        high_frequency_first_difference_weight=0.0,
        high_frequency_second_difference_weight=0.0,
    )
    torch.testing.assert_close(old_total, explicit_zero_total, rtol=0.0, atol=0.0)
    assert old_metrics.high_frequency_first_difference_mae_db_per_bin == 0.0
    assert old_metrics.high_frequency_second_difference_mae_db_per_bin2 == 0.0

    first_weight = 0.25
    second_weight = 0.15
    new_total, new_metrics = calculate_losses(
        prediction_db,
        target_db,
        target_db,
        mca_db,
        directions,
        torch.from_numpy(frequency),
        log_erb_weights,
        target_mean=0.0,
        target_std=1.0,
        erb_weight=0.0,
        high_frequency_weight=0.0,
        ild_weight=0.0,
        direction_weighted_residual=True,
        high_frequency_first_difference_weight=first_weight,
        high_frequency_second_difference_weight=second_weight,
        spectral_difference_minimum_frequency_hz=4000.0,
    )
    expected_increment = (
        first_weight
        * new_metrics.high_frequency_first_difference_mae_db_per_bin
        + second_weight
        * new_metrics.high_frequency_second_difference_mae_db_per_bin2
    )
    torch.testing.assert_close(
        new_total - old_total,
        torch.tensor(expected_increment),
    )
    print(
        {
            "status": "passed",
            "first_difference_mae_db_per_bin": (
                new_metrics.high_frequency_first_difference_mae_db_per_bin
            ),
            "second_difference_mae_db_per_bin2": (
                new_metrics.high_frequency_second_difference_mae_db_per_bin2
            ),
            "objective_increment": expected_increment,
        }
    )


if __name__ == "__main__":
    main()
