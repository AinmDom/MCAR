"""Evaluate a residual MLP against the zero-residual MCA baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization, build_features, list_hdf5_files
from mcar.models.residual_mlp import ResidualMLP, parameter_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--frequencies-per-block", type=int, default=128)
    parser.add_argument(
        "--direction-weighting",
        choices=("uniform", "solid_angle"),
        default="uniform",
    )
    parser.add_argument(
        "--interpolation-only",
        action="store_true",
        help="Evaluate only directions marked by interpolation_evaluation_mask.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation entry point.")
    device = torch.device("cuda")
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    training_arguments = checkpoint["arguments"]
    model = ResidualMLP(
        width=int(training_arguments["width"]),
        block_count=int(training_arguments["block_count"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalization = Normalization.from_json(arguments.dataset_root / "training_statistics.json")
    files = list_hdf5_files(arguments.dataset_root, split=arguments.split)
    use_amp = not arguments.no_amp

    absolute_error = 0.0
    squared_error = 0.0
    baseline_absolute_error = 0.0
    baseline_squared_error = 0.0
    sample_count = 0
    metric_weight_sum = 0.0
    eligible_direction_count: int | None = None
    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            frequency_hz = np.squeeze(handle["frequency_hz"][:])
            directions = np.asarray(
                handle["direction_features"][:],
                dtype=np.float32,
            )
            direction_mask = np.ones(directions.shape[0], dtype=bool)
            if arguments.interpolation_only:
                if "interpolation_evaluation_mask" not in handle:
                    raise ValueError(
                        f"{file_path}: interpolation mask is missing"
                    )
                direction_mask = np.squeeze(
                    handle["interpolation_evaluation_mask"][:]
                ).astype(bool)
            direction_indices = np.flatnonzero(direction_mask)
            if eligible_direction_count is None:
                eligible_direction_count = int(direction_indices.size)
            elif eligible_direction_count != direction_indices.size:
                raise ValueError(
                    f"{file_path}: eligible direction count changed"
                )
            if arguments.direction_weighting == "solid_angle":
                direction_weights = np.asarray(
                    directions[direction_indices, 5],
                    dtype=np.float64,
                )
                if (
                    np.any(direction_weights <= 0.0)
                    or not np.all(np.isfinite(direction_weights))
                ):
                    raise ValueError(
                        f"{file_path}: invalid solid-angle weights"
                    )
            else:
                direction_weights = np.ones(
                    direction_indices.size,
                    dtype=np.float64,
                )
            if arguments.direction_weighting == "solid_angle":
                direction_weights /= np.sum(direction_weights)

            for ear_index in range(2):
                for direction_start in range(
                    0,
                    direction_indices.size,
                    arguments.directions_per_block,
                ):
                    selected_directions = direction_indices[
                        direction_start : (
                            direction_start
                            + arguments.directions_per_block
                        )
                    ]
                    selected_weights = direction_weights[
                        direction_start : (
                            direction_start
                            + arguments.directions_per_block
                        )
                    ]
                    for frequency_start in range(
                        0,
                        frequency_hz.size,
                        arguments.frequencies_per_block,
                    ):
                        frequency_slice = slice(
                            frequency_start,
                            min(
                                frequency_hz.size,
                                frequency_start
                                + arguments.frequencies_per_block,
                            ),
                        )
                        mca = handle["mca_logmag_db"][
                            ear_index,
                            selected_directions,
                            frequency_slice,
                        ]
                        correction = handle["correction_logmag_db"][
                            ear_index,
                            selected_directions,
                            frequency_slice,
                        ]
                        target = handle["target_residual_db"][
                            ear_index,
                            selected_directions,
                            frequency_slice,
                        ]
                        features, target_normalized = build_features(
                            mca,
                            correction,
                            target,
                            directions[selected_directions, :],
                            frequency_hz[frequency_slice],
                            ear_index,
                            normalization,
                        )
                        feature_tensor = torch.from_numpy(features).to(
                            device,
                            non_blocking=True,
                        )
                        target_tensor = torch.from_numpy(
                            target_normalized
                        ).to(device, non_blocking=True)
                        frequency_count = int(
                            frequency_hz[frequency_slice].size
                        )
                        sample_weights = np.repeat(
                            selected_weights,
                            frequency_count,
                        ).reshape(-1, 1)
                        weight_tensor = torch.from_numpy(
                            sample_weights
                        ).to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                        with torch.amp.autocast(
                            "cuda",
                            enabled=use_amp,
                        ):
                            prediction_normalized = model(feature_tensor)
                        prediction_db = (
                            prediction_normalized.float()
                            * normalization.target_std
                            + normalization.target_mean
                        )
                        target_db = (
                            target_tensor.float()
                            * normalization.target_std
                            + normalization.target_mean
                        )
                        error = target_db - prediction_db
                        absolute_error += float(
                            torch.sum(
                                torch.abs(error) * weight_tensor
                            ).item()
                        )
                        squared_error += float(
                            torch.sum(error.square() * weight_tensor).item()
                        )
                        baseline_absolute_error += float(
                            torch.sum(
                                torch.abs(target_db) * weight_tensor
                            ).item()
                        )
                        baseline_squared_error += float(
                            torch.sum(
                                target_db.square() * weight_tensor
                            ).item()
                        )
                        sample_count += target_db.numel()
                        metric_weight_sum += float(
                            torch.sum(weight_tensor).item()
                        )

    metrics = {
        "split": arguments.split,
        "direction_weighting": arguments.direction_weighting,
        "interpolation_only": arguments.interpolation_only,
        "eligible_direction_count": eligible_direction_count,
        "subject_file_count": len(files),
        "sample_count": sample_count,
        "metric_weight_sum": metric_weight_sum,
        "model_mae_db": absolute_error / metric_weight_sum,
        "model_rmse_db": (squared_error / metric_weight_sum) ** 0.5,
        "mca_zero_residual_mae_db": (
            baseline_absolute_error / metric_weight_sum
        ),
        "mca_zero_residual_rmse_db": (
            baseline_squared_error / metric_weight_sum
        )
        ** 0.5,
        "mae_reduction_percent": 100
        * (1 - absolute_error / baseline_absolute_error),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_parameter_count": parameter_count(model),
        "device": torch.cuda.get_device_name(0),
    }
    suffix = arguments.split
    if (
        arguments.interpolation_only
        or arguments.direction_weighting != "uniform"
    ):
        if arguments.interpolation_only:
            suffix += "_interpolation"
        suffix += f"_{arguments.direction_weighting}"
    output_path = arguments.checkpoint.with_name(f"{suffix}_metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
