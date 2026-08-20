"""Small shared residual metrics used during model selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedResidualMetrics:
    residual_mae_db: float
    residual_rmse_db: float
    eligible_direction_count: int
    normalized_direction_weight_sum: float


def solid_angle_weighted_residual_metrics(
    prediction_db: np.ndarray,
    target_db: np.ndarray,
    direction_weights: np.ndarray,
    direction_mask: np.ndarray,
) -> WeightedResidualMetrics:
    """Compute residual MAE/RMSE after renormalizing selected directions.

    Ears and frequencies are equally weighted.  Solid-angle weights apply only
    to the direction axis and are normalized after ``direction_mask`` is
    applied, which is required for interpolation-only evaluation.
    """
    prediction = np.asarray(prediction_db, dtype=np.float64)
    target = np.asarray(target_db, dtype=np.float64)
    weights = np.asarray(direction_weights, dtype=np.float64).reshape(-1)
    mask = np.asarray(direction_mask, dtype=bool).reshape(-1)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction_db and target_db must share [ear,D,F] shape")
    if prediction.shape[0] != 2:
        raise ValueError("residual metrics require exactly two ears")
    if weights.shape != (prediction.shape[1],):
        raise ValueError("direction_weights do not match the direction axis")
    if mask.shape != (prediction.shape[1],):
        raise ValueError("direction_mask does not match the direction axis")
    if not bool(np.any(mask)):
        raise ValueError("direction_mask selects no directions")
    if not bool(np.all(np.isfinite(prediction))) or not bool(
        np.all(np.isfinite(target))
    ):
        raise ValueError("prediction_db and target_db must be finite")
    selected_weights = weights[mask]
    if not bool(np.all(np.isfinite(selected_weights))) or bool(
        np.any(selected_weights <= 0.0)
    ):
        raise ValueError("selected direction weights must be finite and positive")
    selected_weights = selected_weights / np.sum(selected_weights)
    error = prediction[:, mask, :] - target[:, mask, :]
    broadcast_weights = selected_weights.reshape(1, -1, 1)
    mae = np.sum(np.abs(error) * broadcast_weights) / (
        error.shape[0] * error.shape[2]
    )
    mse = np.sum(np.square(error) * broadcast_weights) / (
        error.shape[0] * error.shape[2]
    )
    return WeightedResidualMetrics(
        residual_mae_db=float(mae),
        residual_rmse_db=float(np.sqrt(mse)),
        eligible_direction_count=int(np.count_nonzero(mask)),
        normalized_direction_weight_sum=float(np.sum(selected_weights)),
    )
