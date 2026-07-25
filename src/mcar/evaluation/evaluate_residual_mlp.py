"""Evaluate a residual MLP against the zero-residual MCA baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcar.data import Normalization, iterate_file_blocks, list_hdf5_files
from mcar.models.residual_mlp import ResidualMLP, parameter_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--frequencies-per-block", type=int, default=128)
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
    for file_path in files:
        for features, target_normalized in iterate_file_blocks(
            file_path,
            normalization,
            arguments.directions_per_block,
            arguments.frequencies_per_block,
        ):
            feature_tensor = torch.from_numpy(features).to(device, non_blocking=True)
            target_tensor = torch.from_numpy(target_normalized).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction_normalized = model(feature_tensor)
            prediction_db = prediction_normalized.float() * normalization.target_std + normalization.target_mean
            target_db = target_tensor.float() * normalization.target_std + normalization.target_mean
            error = target_db - prediction_db
            absolute_error += float(torch.sum(torch.abs(error)).item())
            squared_error += float(torch.sum(error.square()).item())
            baseline_absolute_error += float(torch.sum(torch.abs(target_db)).item())
            baseline_squared_error += float(torch.sum(target_db.square()).item())
            sample_count += target_db.numel()

    metrics = {
        "split": arguments.split,
        "subject_file_count": len(files),
        "sample_count": sample_count,
        "model_mae_db": absolute_error / sample_count,
        "model_rmse_db": (squared_error / sample_count) ** 0.5,
        "mca_zero_residual_mae_db": baseline_absolute_error / sample_count,
        "mca_zero_residual_rmse_db": (baseline_squared_error / sample_count) ** 0.5,
        "mae_reduction_percent": 100
        * (1 - absolute_error / baseline_absolute_error),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_parameter_count": parameter_count(model),
        "device": torch.cuda.get_device_name(0),
    }
    output_path = arguments.checkpoint.with_name(f"{arguments.split}_metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
