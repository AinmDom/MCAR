"""Predict complete binaural residual spectra for strict HRTF reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


SCRIPT_ROOT = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_ROOT.parent
RESIDUAL_LEARNING_ROOT = V3_ROOT.parent
SHARED_PYTHON_ROOT = RESIDUAL_LEARNING_ROOT / "python"
for import_root in (SCRIPT_ROOT, SHARED_PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from mlp_cnn_model import ResidualMLPCNN, total_parameter_count  # noqa: E402
from residual_data import Normalization, build_input_features  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("normalization_json", type=Path)
    parser.add_argument("--directions-per-block", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar_attribute(handle: h5py.File, name: str) -> int:
    return int(np.asarray(handle.attrs[name]).item())


def infer_model_architecture(
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, int, int]:
    mlp_width = int(state_dict["mlp.input.0.weight"].shape[0])
    block_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("mlp.blocks.")
    }
    if not block_indices:
        raise ValueError("Checkpoint has no MLP residual blocks")
    mlp_block_count = max(block_indices) + 1
    cnn_channels = int(state_dict["cnn.stem.1.weight"].shape[0])
    return mlp_width, mlp_block_count, cnn_channels


def build_point_features(
    mca_db: np.ndarray,
    correction_db: np.ndarray,
    direction_features: np.ndarray,
    frequency_hz: np.ndarray,
    normalization: Normalization,
) -> np.ndarray:
    direction_count = mca_db.shape[1]
    frequency_count = mca_db.shape[2]
    ear_features = [
        build_input_features(
            mca_db[ear_index],
            correction_db[ear_index],
            direction_features,
            frequency_hz,
            ear_index,
            normalization,
        ).reshape(direction_count, frequency_count, 7)
        for ear_index in range(2)
    ]
    return np.transpose(
        np.stack(ear_features, axis=0),
        (1, 0, 2, 3),
    ).astype(np.float32)


@torch.no_grad()
def predict_file(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model: ResidualMLPCNN,
    normalization: Normalization,
    device: torch.device,
    directions_per_block: int,
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
        _, direction_count, frequency_count = mca_dataset.shape
        directions = np.asarray(source["direction_features"][:], dtype=np.float32)
        frequency_hz = np.squeeze(source["frequency_hz"][:]).astype(np.float32)
        if directions.shape != (direction_count, 6):
            raise ValueError(f"Unexpected direction features {directions.shape}")
        if frequency_hz.shape != (frequency_count,):
            raise ValueError(f"Unexpected frequency vector {frequency_hz.shape}")

        prediction = np.empty(mca_dataset.shape, dtype=np.float32)
        for direction_start in range(0, direction_count, directions_per_block):
            direction_stop = min(
                direction_count, direction_start + directions_per_block
            )
            direction_slice = slice(direction_start, direction_stop)
            mca_db = np.asarray(
                mca_dataset[:, direction_slice, :], dtype=np.float32
            )
            correction_db = np.asarray(
                correction_dataset[:, direction_slice, :], dtype=np.float32
            )
            point_features = build_point_features(
                mca_db,
                correction_db,
                directions[direction_slice],
                frequency_hz,
                normalization,
            )
            point_tensor = torch.from_numpy(point_features).to(
                device, non_blocking=True
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                final_normalized, _, _ = model(point_tensor)
            predicted_db = (
                final_normalized.permute(1, 0, 2).float()
                * normalization.target_std
                + normalization.target_mean
            )
            prediction[:, direction_slice, :] = predicted_db.cpu().numpy()

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
            chunks=(1, min(32, direction_count), frequency_count),
            compression="gzip",
            compression_opts=4,
        )
        destination.attrs["schema_version"] = "1.0"
        destination.attrs["complete"] = 1
        destination.attrs["subject_id"] = subject_id
        destination.attrs["sparse_order"] = sparse_order
        destination.attrs["dense_direction_count"] = dense_direction_count
        destination.attrs["horizontal_direction_count"] = (
            horizontal_direction_count
        )
        destination.attrs["tensor_layout"] = "ear,direction,frequency"
        destination.attrs["target_definition"] = (
            "reference_logmag_db - mca_logmag_db"
        )
        destination.attrs["model"] = "ResidualMLPCNN"
        destination.attrs["checkpoint"] = str(checkpoint_path.resolve())
    temporary_path.replace(output_path)
    elapsed = time.perf_counter() - started
    return {
        "subject_id": subject_id,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "direction_count": direction_count,
        "frequency_count": frequency_count,
        "prediction_min_db": float(np.min(prediction)),
        "prediction_max_db": float(np.max(prediction)),
        "prediction_mean_db": float(np.mean(prediction)),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    arguments = parse_arguments()
    if arguments.directions_per_block < 1:
        raise ValueError("directions_per_block must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v3 reconstruction inference")

    device = torch.device("cuda")
    use_amp = not arguments.no_amp
    checkpoint = torch.load(
        arguments.checkpoint, map_location=device, weights_only=False
    )
    state_dict = checkpoint["model_state"]
    mlp_width, mlp_block_count, cnn_channels = infer_model_architecture(
        state_dict
    )
    model = ResidualMLPCNN(
        mlp_width=mlp_width,
        mlp_block_count=mlp_block_count,
        cnn_channels=cnn_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    normalization = Normalization.from_json(arguments.normalization_json)

    input_paths = sorted(arguments.input_root.glob("subjects/pp*/model_input.h5"))
    if not input_paths:
        raise FileNotFoundError(
            f"No subjects/pp*/model_input.h5 under {arguments.input_root}"
        )
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, input_path in enumerate(input_paths, start=1):
        output_path = (
            arguments.output_root
            / "subjects"
            / input_path.parent.name
            / "predicted_residual.h5"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not arguments.overwrite:
            with h5py.File(output_path, "r") as existing:
                if int(np.asarray(existing.attrs.get("complete", 0)).item()) == 1:
                    print(f"[{index:02d}/{len(input_paths):02d}] skip {output_path}")
                    continue
        report = predict_file(
            input_path,
            output_path,
            arguments.checkpoint,
            model,
            normalization,
            device,
            arguments.directions_per_block,
            use_amp,
        )
        reports.append(report)
        print(
            f"[{index:02d}/{len(input_paths):02d}] pp{report['subject_id']} "
            f"{report['elapsed_seconds']:.2f}s "
            f"[{report['prediction_min_db']:.2f}, "
            f"{report['prediction_max_db']:.2f}] dB"
        )

    summary = {
        "status": "completed",
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "normalization_json": str(arguments.normalization_json.resolve()),
        "input_root": str(arguments.input_root.resolve()),
        "output_root": str(arguments.output_root.resolve()),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "amp": use_amp,
        "directions_per_block": arguments.directions_per_block,
        "model_parameter_count": total_parameter_count(model),
        "processed_subject_count": len(reports),
        "elapsed_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        ),
        "subjects": reports,
    }
    report_path = arguments.output_root / "inference_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"output={report_path}")


if __name__ == "__main__":
    main()
