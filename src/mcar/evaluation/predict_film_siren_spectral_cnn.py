"""Predict validation residuals from one Stage-D spectral-refiner checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mcar.paths import project_root
from mcar.predictors import FilmSirenSpectralCNNPredictor
from mcar.training.train_film_siren import file_sha256, split_subject_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--subject-limit", type=int)
    return parser.parse_args()


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def write_prediction(
    source_path: Path,
    output_path: Path,
    prediction: np.ndarray,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    with h5py.File(source_path, "r") as source:
        subject_id = int(np.asarray(source.attrs["subject_id"]).item())
        subject_label = decode_attribute(source.attrs["subject_label"])
        split = decode_attribute(source.attrs["split"])
    if split != "val":
        raise PermissionError("Stage-D development inference permits validation only")
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary, "w") as destination:
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
        destination.attrs["model_family"] = "FilmSirenSpectralCNN"
        destination.attrs["checkpoint"] = str(checkpoint_path)
        destination.attrs["checkpoint_sha256"] = checkpoint_sha256
    temporary.replace(output_path)
    return {
        "subject_id": subject_id,
        "subject_label": subject_label,
        "output": str(output_path),
        "shape": list(prediction.shape),
        "finite": bool(np.all(np.isfinite(prediction))),
    }


def main() -> None:
    arguments = parse_arguments()
    if arguments.directions_per_block < 1:
        raise ValueError("directions-per-block must be positive")
    if arguments.subject_limit is not None and arguments.subject_limit < 1:
        raise ValueError("subject-limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage-D validation inference")
    root = project_root()
    checkpoint_path = arguments.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_stage") != "film_siren_spectral_cnn_stage_d":
        raise ValueError("Checkpoint is not a Stage-D spectral-refiner model")
    experiment = checkpoint["experiment_configuration"]
    dataset_root = (root / experiment["dataset_root"]).resolve()
    split_csv = (root / experiment["subject_split_csv"]).resolve()
    q26_csv = (root / experiment["q26_csv"]).resolve()
    q26_normalization = (root / experiment["q26_normalization"]).resolve()
    subject_rows = split_subject_paths(dataset_root, split_csv, "val")
    if arguments.subject_limit is not None:
        subject_rows = subject_rows[: arguments.subject_limit]
    checkpoint_sha256 = file_sha256(checkpoint_path).upper()
    predictor = FilmSirenSpectralCNNPredictor(
        checkpoint_path,
        split_csv,
        q26_csv,
        q26_normalization,
        device=torch.device("cuda"),
        directions_per_block=arguments.directions_per_block,
        allow_test=False,
    )
    output_root = root / "artifacts" / "reconstruction" / arguments.run_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index, (_, subject_label, source_path) in enumerate(subject_rows, start=1):
        prediction = predictor.predict_residual_db(source_path)
        if not bool(np.all(np.isfinite(prediction))):
            raise FloatingPointError(f"Non-finite prediction for {subject_label}")
        output_path = output_root / "subjects" / subject_label / "prediction.h5"
        rows.append(
            write_prediction(
                source_path,
                output_path,
                prediction,
                checkpoint_path,
                checkpoint_sha256,
            )
        )
        print(f"predicted [{index}/{len(subject_rows)}] {subject_label}", flush=True)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "val",
        "subject_count": len(subject_rows),
        "test_subject_count_read": 0,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "run_name": arguments.run_name,
        "output_root": str(output_root.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "subjects": rows,
    }
    (output_root / "inference_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
