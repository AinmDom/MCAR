"""Predict full-spectrum SONICOM residuals with a locked MLP+CNN model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization, list_hdf5_files
from mcar.evaluation.evaluate_mlp_cnn_v3 import (
    build_point_features,
    infer_global_context_arguments,
    infer_model_architecture,
)
from mcar.models.residual_mlp_cnn import ResidualMLPCNN, total_parameter_count
from mcar.paths import project_root


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--subject-limit", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


@torch.no_grad()
def predict_subject(
    source_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model: ResidualMLPCNN,
    normalization: Normalization,
    device: torch.device,
    directions_per_block: int,
    use_amp: bool,
    expected_split: str,
) -> dict[str, object]:
    started = time.perf_counter()
    with h5py.File(source_path, "r") as source:
        split = decode_attribute(source.attrs["split"])
        if split != expected_split:
            raise ValueError(
                f"Expected split {expected_split!r} in {source_path}, found {split!r}"
            )
        subject_id = int(np.asarray(source.attrs["subject_id"]).item())
        subject_label = decode_attribute(source.attrs["subject_label"])
        mca = source["mca_logmag_db"]
        correction = source["correction_logmag_db"]
        if mca.shape != correction.shape or len(mca.shape) != 3:
            raise ValueError(f"Unexpected spectral layout in {source_path}")
        if mca.shape[0] != 2:
            raise ValueError(f"Expected two ears in {source_path}")
        directions = np.asarray(source["direction_features"][:], dtype=np.float32)
        frequency_hz = np.squeeze(source["frequency_hz"][:]).astype(np.float32)
        if directions.shape != (mca.shape[1], 6):
            raise ValueError(f"Unexpected direction features in {source_path}")
        if frequency_hz.shape != (mca.shape[2],):
            raise ValueError(f"Unexpected frequency vector in {source_path}")

        prediction = np.empty(mca.shape, dtype=np.float32)
        for direction_start in range(0, mca.shape[1], directions_per_block):
            direction_stop = min(
                mca.shape[1], direction_start + directions_per_block
            )
            direction_slice = slice(direction_start, direction_stop)
            mca_block = np.asarray(mca[:, direction_slice, :], dtype=np.float32)
            correction_block = np.asarray(
                correction[:, direction_slice, :], dtype=np.float32
            )
            point_features = build_point_features(
                mca_block,
                correction_block,
                directions[direction_slice, :],
                frequency_hz,
                normalization,
            )
            point_tensor = torch.from_numpy(point_features).to(
                device, non_blocking=True
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                normalized_prediction, _, _ = model(point_tensor)
            predicted_db = (
                normalized_prediction.permute(1, 0, 2).float()
                * normalization.target_std
                + normalization.target_mean
            )
            prediction[:, direction_slice, :] = predicted_db.cpu().numpy()

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary_path, "w") as destination:
        destination.create_dataset(
            "predicted_residual_db",
            data=prediction,
            chunks=(1, min(64, prediction.shape[1]), prediction.shape[2]),
            compression="gzip",
            compression_opts=4,
        )
        destination.attrs["schema_version"] = "1.0"
        destination.attrs["complete"] = 1
        destination.attrs["subject_id"] = subject_id
        destination.attrs["subject_label"] = subject_label
        destination.attrs["split"] = split
        destination.attrs["tensor_layout"] = "ear,direction,frequency"
        destination.attrs["model_family"] = "ResidualMLPCNN"
        destination.attrs["checkpoint"] = str(checkpoint_path.resolve())
        destination.attrs["source_hdf5"] = str(source_path.resolve())
    temporary_path.replace(output_path)

    return {
        "subject_id": subject_id,
        "subject_label": subject_label,
        "output": str(output_path),
        "shape": list(prediction.shape),
        "minimum_db": float(np.min(prediction)),
        "maximum_db": float(np.max(prediction)),
        "mean_db": float(np.mean(prediction)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SONICOM reconstruction inference")
    if arguments.directions_per_block < 1:
        raise ValueError("directions-per-block must be positive")
    if arguments.subject_limit is not None and arguments.subject_limit < 1:
        raise ValueError("subject-limit must be positive")
    if arguments.split == "test" and not arguments.allow_test:
        raise ValueError(
            "The locked SONICOM test split requires explicit --allow-test"
        )

    subject_files = list_hdf5_files(
        arguments.dataset_root, split=arguments.split
    )
    if len(subject_files) != 44:
        raise ValueError(
            f"Expected 44 {arguments.split} files, found {len(subject_files)}"
        )
    if arguments.subject_limit is not None:
        subject_files = subject_files[: arguments.subject_limit]

    device = torch.device("cuda")
    checkpoint = torch.load(
        arguments.checkpoint, map_location=device, weights_only=False
    )
    state_dict = checkpoint["model_state"]
    mlp_width, mlp_block_count, cnn_channels = infer_model_architecture(state_dict)
    model = ResidualMLPCNN(
        mlp_width=mlp_width,
        mlp_block_count=mlp_block_count,
        cnn_channels=cnn_channels,
        **infer_global_context_arguments(checkpoint),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    normalization = Normalization.from_json(
        arguments.dataset_root / "training_statistics.json"
    )
    output_root = (
        project_root() / "artifacts" / "reconstruction" / arguments.run_name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    use_amp = not arguments.no_amp

    started = time.perf_counter()
    reports: list[dict[str, object]] = []
    for index, source_path in enumerate(subject_files, start=1):
        with h5py.File(source_path, "r") as source:
            subject_label = decode_attribute(source.attrs["subject_label"])
        output_path = output_root / "subjects" / subject_label / "prediction.h5"
        if output_path.exists() and not arguments.overwrite:
            with h5py.File(output_path, "r") as existing:
                complete = int(np.asarray(existing.attrs.get("complete", 0)).item())
                existing_checkpoint = decode_attribute(
                    existing.attrs.get("checkpoint", "")
                )
            if complete == 1 and existing_checkpoint == str(
                arguments.checkpoint.resolve()
            ):
                print(f"skip complete [{index}/{len(subject_files)}]: {subject_label}")
                continue
        report = predict_subject(
            source_path,
            output_path,
            arguments.checkpoint,
            model,
            normalization,
            device,
            arguments.directions_per_block,
            use_amp,
            arguments.split,
        )
        reports.append(report)
        print(
            f"predicted [{index}/{len(subject_files)}] {subject_label} "
            f"in {report['elapsed_seconds']:.2f}s"
        )

    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "split": arguments.split,
        "test_subject_count_read": (
            len(subject_files) if arguments.split == "test" else 0
        ),
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_stage": checkpoint.get("training_stage", "unknown"),
        "dataset_root": str(arguments.dataset_root.resolve()),
        "run_name": arguments.run_name,
        "output_root": str(output_root.resolve()),
        "subject_count": len(subject_files),
        "validation_subject_count": (
            len(subject_files) if arguments.split == "val" else 0
        ),
        "newly_processed_subject_count": len(reports),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "amp": use_amp,
        "model_parameter_count": total_parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
        "subjects": reports,
    }
    report_path = output_root / "inference_report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output={report_path}")


if __name__ == "__main__":
    main()
