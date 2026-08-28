"""Predict a hash-locked validation ensemble of Stage-D spectral CNNs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from mcar.evaluation.predict_film_siren_spectral_cnn import write_prediction
from mcar.paths import project_root
from mcar.predictors import FilmSirenSpectralCNNPredictor
from mcar.training.train_film_siren import file_sha256, split_subject_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--subject-limit", type=int)
    return parser.parse_args()


def manifest_identity(payload: dict) -> str:
    values = dict(payload)
    claimed = values.pop("identity_sha256")
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Manifest identity mismatch")
    return actual


def main() -> None:
    arguments = parse_arguments()
    if arguments.directions_per_block < 1:
        raise ValueError("directions-per-block must be positive")
    if arguments.subject_limit is not None and arguments.subject_limit < 1:
        raise ValueError("subject-limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ensemble validation inference")

    root = project_root()
    manifest_path = arguments.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest_identity(manifest)
    if manifest["status"] != "frozen" or manifest["e_final"] != 190:
        raise ValueError("Manifest is not the frozen formal E190 ensemble")
    if manifest["dataset"]["split"] != "val":
        raise PermissionError("Formal Stage-D inference permits validation only")
    weights = np.asarray(manifest["ensemble"]["weights"], dtype=np.float64)
    if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Expected three ensemble weights summing to one")

    dataset_root = (root / manifest["dataset"]["root"]).resolve()
    split_csv = (root / manifest["dataset"]["split_csv"]).resolve()
    q26_csv = (root / manifest["dataset"]["q26_csv"]).resolve()
    q26_normalization = (root / manifest["dataset"]["q26_normalization"]).resolve()
    predictors = []
    for member in manifest["members"]:
        config_path = root / member["config"]
        checkpoint_path = root / member["checkpoint"]
        if file_sha256(config_path).upper() != member["config_sha256"]:
            raise ValueError(f"Config hash mismatch for seed {member['seed']}")
        if file_sha256(checkpoint_path).upper() != member["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for seed {member['seed']}")
        predictors.append(
            FilmSirenSpectralCNNPredictor(
                checkpoint_path,
                split_csv,
                q26_csv,
                q26_normalization,
                device=torch.device("cuda"),
                directions_per_block=arguments.directions_per_block,
                allow_test=False,
            )
        )

    subject_rows = split_subject_paths(dataset_root, split_csv, "val")
    if arguments.subject_limit is not None:
        subject_rows = subject_rows[: arguments.subject_limit]
    output_root = root / "artifacts" / "reconstruction" / arguments.run_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")

    started = time.perf_counter()
    rows = []
    for index, (_, subject_label, source_path) in enumerate(subject_rows, start=1):
        member_predictions = [p.predict_residual_db(source_path) for p in predictors]
        prediction = np.average(
            np.stack(member_predictions, axis=0), axis=0, weights=weights
        ).astype(np.float32)
        if not bool(np.all(np.isfinite(prediction))):
            raise FloatingPointError(f"Non-finite prediction for {subject_label}")
        output_path = output_root / "subjects" / subject_label / "prediction.h5"
        rows.append(
            write_prediction(
                source_path,
                output_path,
                prediction,
                manifest_path,
                identity,
                model_family="FilmSirenSpectralCNNEnsemble",
            )
        )
        print(f"predicted [{index}/{len(subject_rows)}] {subject_label}", flush=True)

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "val",
        "subject_count": len(subject_rows),
        "test_subject_count_read": 0,
        "manifest": str(manifest_path),
        "manifest_identity_sha256": identity,
        "ensemble": manifest["ensemble"],
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
