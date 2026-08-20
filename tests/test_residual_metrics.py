from __future__ import annotations

import numpy as np
import pytest

from mcar.evaluation.residual_metrics import (
    solid_angle_weighted_residual_metrics,
)


def test_constant_error_is_invariant_to_weights_and_mask() -> None:
    target = np.zeros((2, 4, 3), dtype=np.float32)
    prediction = np.full_like(target, 2.5)
    metrics = solid_angle_weighted_residual_metrics(
        prediction,
        target,
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([True, False, True, False]),
    )
    assert metrics.residual_mae_db == pytest.approx(2.5)
    assert metrics.residual_rmse_db == pytest.approx(2.5)
    assert metrics.eligible_direction_count == 2
    assert metrics.normalized_direction_weight_sum == pytest.approx(1.0)


def test_selected_weights_are_renormalized() -> None:
    target = np.zeros((2, 3, 1), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[:, 0, :] = 1.0
    prediction[:, 1, :] = 3.0
    metrics = solid_angle_weighted_residual_metrics(
        prediction,
        target,
        np.asarray([2.0, 1.0, 100.0]),
        np.asarray([True, True, False]),
    )
    assert metrics.residual_mae_db == pytest.approx(5.0 / 3.0)
    assert metrics.residual_rmse_db == pytest.approx(np.sqrt(11.0 / 3.0))


@pytest.mark.parametrize(
    ("weights", "mask", "message"),
    [
        ([1.0, 1.0], [False, False], "selects no directions"),
        ([1.0, 0.0], [True, True], "finite and positive"),
    ],
)
def test_invalid_selection_is_rejected(
    weights: list[float], mask: list[bool], message: str
) -> None:
    values = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        solid_angle_weighted_residual_metrics(
            values,
            values,
            np.asarray(weights),
            np.asarray(mask),
        )
