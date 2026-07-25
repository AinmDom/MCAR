"""Predict residuals for cached MCA reconstruction inputs on the GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization, build_input_features
from mcar.models.residual_mlp import ResidualMLP, parameter_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reconstruction_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("normalization_json", type=Path)
    parser.add_argument(
        "--input-root",
        type=Path,
        help=(
            "Optional reconstruction root containing model_input.h5 files. "
            "Predictions are still written under reconstruction_root."
        ),
    )
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--frequencies-per-block", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar_attribute(handle: h5py.File, name: str) -> int:
    return int(np.asarray(handle.attrs[name]).item())


@torch.no_grad()
def predict_file(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model: ResidualMLP,
    normalization: Normalization,
    device: torch.device,
    directions_per_block: int,
    frequencies_per_block: int,
    use_amp: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    with h5py.File(input_path, "r") as source:
        mca_dataset = source["mca_logmag_db"]
        correction_dataset = source["correction_logmag_db"]
        if mca_dataset.shape != correction_dataset.shape:
            raise ValueError(f"Input tensor mismatch in {input_path}")
        if len(mca_dataset.shape) != 3 or mca_dataset.shape[0] != 2:
            raise ValueError(
                f"Expected [ear,direction,frequency], got {mca_dataset.shape}"
            )
        ear_count, direction_count, frequency_count = mca_dataset.shape
        directions = source["direction_features"][:]
        frequency_hz = np.squeeze(source["frequency_hz"][:])
        if directions.shape != (direction_count, 6):
            raise ValueError(f"Unexpected direction features {directions.shape}")
        if frequency_hz.shape != (frequency_count,):
            raise ValueError(f"Unexpected frequency vector {frequency_hz.shape}")

        prediction = np.empty(mca_dataset.shape, dtype=np.float32)
        for ear_index in range(ear_count):
            for direction_start in range(0, direction_count, directions_per_block):
                direction_stop = min(
                    direction_count, direction_start + directions_per_block
                )
                direction_slice = slice(direction_start, direction_stop)
                for frequency_start in range(
                    0, frequency_count, frequencies_per_block
                ):
                    frequency_stop = min(
                        frequency_count,
                        frequency_start + frequencies_per_block,
                    )
                    frequency_slice = slice(frequency_start, frequency_stop)
                    mca = mca_dataset[ear_index, direction_slice, frequency_slice]
                    correction = correction_dataset[
                        ear_index, direction_slice, frequency_slice
                    ]
                    features = build_input_features(
                        mca,
                        correction,
                        directions[direction_slice, :],
                        frequency_hz[frequency_slice],
                        ear_index,
                        normalization,
                    )
                    feature_tensor = torch.from_numpy(features).to(
                        device, non_blocking=True
                    )
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        normalized = model(feature_tensor)
                    predicted_db = (
                        normalized.float() * normalization.target_std
                        + normalization.target_mean
                    )
                    prediction[ear_index, direction_slice, frequency_slice] = (
                        predicted_db.reshape(
                            direction_stop - direction_start,
                            frequency_stop - frequency_start,
                        )
                        .cpu()
                        .numpy()
                    )

        subject_id = scalar_attribute(source, "subject_id")
        sparse_order = scalar_attribute(source, "sparse_order")
        dense_direction_count = scalar_attribute(source, "dense_direction_count")
        horizontal_direction_count = scalar_attribute(
            source, "horizontal_direction_count"
        )

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_path.unlink(missing_ok=True)
    with h5py.File(temporary_path, "w") as destination:
        destination.create_dataset(
            "predicted_residual_db",
            data=prediction,
            chunks=(1, min(64, direction_count), min(128, frequency_count)),
            compression="gzip",
            compression_opts=4,
        )
        destination.attrs["schema_version"] = "1.0"
        destination.attrs["complete"] = 1
        destination.attrs["subject_id"] = subject_id
        destination.attrs["sparse_order"] = sparse_order
        destination.attrs["dense_direction_count"] = dense_direction_count
        destination.attrs["horizontal_direction_count"] = horizontal_direction_count
        destination.attrs["tensor_layout"] = "ear,direction,frequency"
        destination.attrs["target_definition"] = (
            "reference_logmag_db - mca_logmag_db"
        )
        destination.attrs["checkpoint"] = str(checkpoint_path.resolve())
    temporary_path.replace(output_path)
    elapsed = time.perf_counter() - started
    return {
        "subject_id": subject_id,
        "input": str(input_path),
        "output": str(output_path),
        "direction_count": direction_count,
        "frequency_count": frequency_count,
        "prediction_min_db": float(np.min(prediction)),
        "prediction_max_db": float(np.max(prediction)),
        "prediction_mean_db": float(np.mean(prediction)),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reconstruction inference.")
    if arguments.directions_per_block < 1 or arguments.frequencies_per_block < 1:
        raise ValueError("Block dimensions must be positive")

    device = torch.device("cuda")
    checkpoint = torch.load(
        arguments.checkpoint, map_location=device, weights_only=False
    )
    training_arguments = checkpoint["arguments"]
    model = ResidualMLP(
        width=int(training_arguments["width"]),
        block_count=int(training_arguments["block_count"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalization = Normalization.from_json(arguments.normalization_json)
    input_root = (
        arguments.input_root
        if arguments.input_root is not None
        else arguments.reconstruction_root
    )
    arguments.reconstruction_root.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(
        input_root.glob("subjects/pp*/model_input.h5")
    )
    if not input_paths:
        raise FileNotFoundError(
            f"No subjects/pp*/model_input.h5 under "
            f"{input_root}"
        )

    reports: list[dict[str, object]] = []
    for input_path in input_paths:
        output_path = (
            arguments.reconstruction_root
            / "subjects"
            / input_path.parent.name
            / "predicted_residual.h5"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not arguments.overwrite:
            with h5py.File(output_path, "r") as existing:
                if int(np.asarray(existing.attrs.get("complete", 0)).item()) == 1:
                    print(f"skip complete: {output_path}")
                    continue
        report = predict_file(
            input_path,
            output_path,
            arguments.checkpoint,
            model,
            normalization,
            device,
            arguments.directions_per_block,
            arguments.frequencies_per_block,
            not arguments.no_amp,
        )
        reports.append(report)
        print(json.dumps(report))

    summary = {
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "normalization_json": str(arguments.normalization_json.resolve()),
        "input_root": str(input_root.resolve()),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "amp": not arguments.no_amp,
        "model_parameter_count": parameter_count(model),
        "processed_subject_count": len(reports),
        "subjects": reports,
    }
    output_path = arguments.reconstruction_root / "inference_report.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
