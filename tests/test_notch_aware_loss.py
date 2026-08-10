"""CPU checks for the differentiable multi-scale notch-depth objective."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mcar.losses import multi_scale_notch_depth_mae
from mcar.training.train_mlp_cnn_v3 import training_stage
from mcar.training.train_mlp_v2 import calculate_losses


def main() -> None:
    frequency_hz = torch.linspace(3000.0, 19000.0, 41)
    direction_weights = torch.tensor([0.75, 0.25])
    reference_db = torch.zeros(2, 2, 41)
    reference_db[:, 0, 20] = -8.0

    identical = multi_scale_notch_depth_mae(
        reference_db,
        reference_db,
        direction_weights,
        frequency_hz,
        4000.0,
        18000.0,
        (2, 4),
        1.0,
        0.5,
    )
    torch.testing.assert_close(identical, torch.tensor(0.0), atol=0.0, rtol=0.0)

    corrected_db = torch.zeros(2, 2, 41, requires_grad=True)
    missing_notch = multi_scale_notch_depth_mae(
        corrected_db,
        reference_db,
        direction_weights,
        frequency_hz,
        4000.0,
        18000.0,
        (2, 4),
        1.0,
        0.5,
    )
    if not bool(missing_notch > 0.0):
        raise AssertionError("A missing reference notch was not penalized")
    missing_notch.backward()
    if corrected_db.grad is None or not bool(
        torch.all(torch.isfinite(corrected_db.grad))
    ):
        raise AssertionError("Notch-aware backward produced invalid gradients")
    if not bool(torch.any(corrected_db.grad != 0.0)):
        raise AssertionError("A missing notch did not receive a useful gradient")

    false_notch_db = torch.zeros(2, 2, 41)
    false_notch_db[:, 1, 25] = -6.0
    false_notch = multi_scale_notch_depth_mae(
        false_notch_db,
        torch.zeros_like(false_notch_db),
        torch.tensor([0.0, 1.0]),
        frequency_hz,
        4000.0,
        18000.0,
        (2, 4),
        1.0,
        0.5,
    )
    ignored_false_notch = multi_scale_notch_depth_mae(
        false_notch_db,
        torch.zeros_like(false_notch_db),
        torch.tensor([1.0, 0.0]),
        frequency_hz,
        4000.0,
        18000.0,
        (2, 4),
        1.0,
        0.5,
    )
    if not bool(false_notch > 0.0):
        raise AssertionError("A false predicted notch was not penalized")
    torch.testing.assert_close(
        ignored_false_notch, torch.tensor(0.0), atol=0.0, rtol=0.0
    )

    direction_count = 2
    target_std = 2.0
    prediction_normalized = torch.zeros(2, direction_count, 41)
    target_db = reference_db.clone()
    target_normalized = target_db / target_std
    mca_db = torch.zeros_like(target_db)
    direction_features = torch.zeros(direction_count, 6)
    direction_features[:, 3] = torch.tensor([-1.0, 1.0])
    direction_features[:, 5] = direction_weights
    erb_weights = torch.full((5, 41), 1.0 / 41.0)
    log_erb_weights = torch.log(erb_weights).view(1, 1, 5, 41)
    old_total, _ = calculate_losses(
        prediction_normalized,
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
        log_erb_weights,
        0.0,
        target_std,
        0.0,
        0.0,
        0.0,
        True,
    )
    new_total, metrics = calculate_losses(
        prediction_normalized,
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
        log_erb_weights,
        0.0,
        target_std,
        0.0,
        0.0,
        0.0,
        True,
        notch_depth_weight=0.3,
        notch_minimum_frequency_hz=4000.0,
        notch_maximum_frequency_hz=18000.0,
        notch_radii_bins=(2, 4),
        notch_depth_threshold_db=1.0,
        notch_softplus_temperature_db=0.5,
    )
    expected_increment = 0.3 * metrics.notch_depth_mae_db / target_std
    torch.testing.assert_close(
        new_total - old_total,
        torch.tensor(expected_increment),
        atol=1e-6,
        rtol=1e-5,
    )

    stage = training_stage(
        SimpleNamespace(
            high_frequency_first_difference_weight=0.0,
            high_frequency_second_difference_weight=0.0,
            spectral_band_ild_weight=0.0,
            notch_depth_weight=0.3,
            unfreeze_mlp=False,
            dual_sampling_strict_ild=True,
            global_context_attention=False,
            freeze_local_cnn=False,
            ild_loss_mode="strict_hrir",
            horizontal_only=False,
        )
    )
    if not stage.endswith("notch_aware_v34c"):
        raise AssertionError(f"Unexpected v3.4c training stage: {stage}")
    print(
        {
            "status": "passed",
            "notch_depth_mae_db": metrics.notch_depth_mae_db,
            "objective_increment": float((new_total - old_total).item()),
            "training_stage": stage,
        }
    )


if __name__ == "__main__":
    main()
