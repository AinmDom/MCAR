"""Predict validation residuals from one Stage-E bounded correction model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mcar.evaluation.predict_film_siren_spectral_cnn import write_prediction
from mcar.paths import project_root
from mcar.predictors import BoundedMcarFilmCorrectionPredictor
from mcar.training.train_film_siren import file_sha256, split_subject_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--directions-per-block", type=int, default=32)
    parser.add_argument("--subject-limit", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage-E validation inference")
    if arguments.directions_per_block < 1:
        raise ValueError("directions-per-block must be positive")
    root = project_root()
    checkpoint_path = arguments.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("training_stage") != "bounded_mcar_film_correction_stage_e":
        raise ValueError("Checkpoint is not a Stage-E bounded correction model")
    experiment = payload["experiment_configuration"]
    dataset_root = (root / experiment["dataset_root"]).resolve()
    split_csv = (root / experiment["subject_split_csv"]).resolve()
    q26_csv = (root / experiment["q26_csv"]).resolve()
    q26_normalization = (root / experiment["q26_normalization"]).resolve()
    subjects = split_subject_paths(dataset_root, split_csv, "val")
    if arguments.subject_limit is not None:
        subjects = subjects[: arguments.subject_limit]
    checkpoint_sha256 = file_sha256(checkpoint_path).upper()
    predictor = BoundedMcarFilmCorrectionPredictor(
        checkpoint_path,
        split_csv,
        q26_csv,
        q26_normalization,
        device=torch.device("cuda"),
        directions_per_block=arguments.directions_per_block,
        allow_test=False,
    )
    output_root = root / "artifacts/reconstruction" / arguments.run_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    started = time.perf_counter()
    rows = []
    for index, (_, subject_label, source_path) in enumerate(subjects, start=1):
        prediction = predictor.predict_residual_db(source_path)
        if not bool(np.all(np.isfinite(prediction))):
            raise FloatingPointError(f"Non-finite prediction for {subject_label}")
        rows.append(
            write_prediction(
                source_path,
                output_root / "subjects" / subject_label / "prediction.h5",
                prediction,
                checkpoint_path,
                checkpoint_sha256,
                model_family="BoundedMcarFilmCorrection",
            )
        )
        print(f"predicted [{index}/{len(subjects)}] {subject_label}", flush=True)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "val",
        "subject_count": len(subjects),
        "test_subject_count_read": 0,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "run_name": arguments.run_name,
        "output_root": str(output_root.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "subjects": rows,
    }
    (output_root / "inference_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
