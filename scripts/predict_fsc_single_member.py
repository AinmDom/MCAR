"""Predict one frozen FSC E190 member on validation or explicitly authorized test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mcar.paths import project_root
from mcar.predictors import FilmSirenSpectralCNNPredictor
from mcar.q26_condition import sparse_source_indices
from mcar.fsp_ae_data import q26_source_indices
from mcar.training.train_film_siren import file_sha256


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", type=Path)
    parser.add_argument("direction_count", type=int, choices=(14, 26, 50))
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--directions-per-block", type=int, default=64)
    return parser.parse_args()


def decode(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def subject_paths(split_csv: Path, dataset_root: Path, split: str, count: int) -> list[tuple[int, str, Path]]:
    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if len(rows) != 44:
        raise ValueError(f"Expected 44 {split} subjects, found {len(rows)}")
    items = []
    for row in rows:
        label = row["subject_id"]
        path = dataset_root / "subjects" / label / f"q{count}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        items.append((int(label[1:]), label, path))
    return items


def write_prediction(source_path: Path, output_path: Path, prediction: np.ndarray, provenance: dict[str, Any]) -> dict[str, Any]:
    with h5py.File(source_path, "r") as source:
        subject_id = int(np.asarray(source.attrs["subject_id"]).item())
        subject_label = decode(source.attrs["subject_label"])
        split = decode(source.attrs["split"])
    if prediction.shape != (2, 793, 463) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError(f"Invalid prediction for {subject_label}")
    temporary = output_path.with_suffix(".h5.partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary, "w") as destination:
        destination.create_dataset(
            "predicted_residual_db", data=prediction.astype(np.float32),
            chunks=(1, min(64, prediction.shape[1]), prediction.shape[2]),
            compression="gzip", compression_opts=4,
        )
        destination.attrs.update({
            "schema_version": "1.0", "complete": 1,
            "subject_id": subject_id, "subject_label": subject_label, "split": split,
            "tensor_layout": "ear,direction,frequency",
            "model_family": "FilmSirenSpectralCNNSingleMember",
            "checkpoint": provenance["checkpoint"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "selected_q26_seed": int(provenance["selected_q26_seed"]),
            "member_identity": provenance["member_identity"],
            "selection_rule": provenance["selection_rule"],
        })
    temporary.replace(output_path)
    return {"subject_id": subject_id, "subject_label": subject_label, "shape": list(prediction.shape), "finite": True}


def main() -> None:
    args = parse_arguments()
    if args.split == "test" and not args.allow_test:
        raise PermissionError("Test prediction requires --allow-test")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FSC prediction")
    root = project_root()
    config_path = args.configuration.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError("Frozen configuration required")
    if args.split == "test" and not config["authorization"].get("test_access_allowed"):
        raise PermissionError("Configuration does not authorize test access")
    run = next(item for item in config["runs"] if int(item["direction_count"]) == args.direction_count)
    checkpoint = root / run["checkpoint"]
    model_config = root / run["config"]
    report_path = root / run["training_report"]
    if file_sha256(checkpoint).upper() != run["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    if file_sha256(model_config).upper() != run["config_sha256"]:
        raise ValueError(f"Configuration hash mismatch: {model_config}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("authoritative_checkpoint") != "last.pt" or int(report.get("test_subjects_read", -1)) != 0:
        raise ValueError(f"Unsafe training provenance: {report_path}")
    if int(report.get("seed", -1)) != int(run["training_rng_seed"]):
        raise ValueError("Training RNG seed mismatch")
    dataset_key = "dataset_root_test" if args.split == "test" else "dataset_root_trainval"
    dataset_root = (root / run[dataset_key]).resolve()
    split_csv = (root / config["split_csv"]).resolve()
    sparse_grid = (root / run["sparse_grid_csv"]).resolve()
    normalization = (root / run["normalization"]).resolve()
    output_name = run[f"output_{'test' if args.split == 'test' else 'validation'}"]
    output_root = root / "artifacts" / "reconstruction" / output_name
    if output_root.exists():
        raise FileExistsError(output_root)
    indices = (
        q26_source_indices(sparse_grid)
        if args.direction_count == 26
        else sparse_source_indices(sparse_grid, args.direction_count)
    )
    predictor = FilmSirenSpectralCNNPredictor(
        checkpoint, split_csv, sparse_grid, normalization,
        device=torch.device("cuda"), directions_per_block=args.directions_per_block,
        allow_test=args.split == "test", condition_source_indices=indices,
        condition_dataset_root=dataset_root, condition_filename=f"q{args.direction_count}.h5",
    )
    provenance = {
        "configuration": str(config_path), "checkpoint": run["checkpoint"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "selected_q26_seed": config["selection"]["selected_q26_seed"],
        "member_identity": run["member_identity"],
        "selection_rule": config["selection"]["rule"],
    }
    started = time.perf_counter()
    rows = []
    for index, (_, label, source) in enumerate(subject_paths(split_csv, dataset_root, args.split, args.direction_count), start=1):
        prediction = predictor.predict_residual_db(source)
        rows.append(write_prediction(source, output_root / "subjects" / label / "prediction.h5", prediction, provenance))
        print(f"{args.split} single-member predicted [{index}/44] {label}", flush=True)
    report_payload = {
        "schema_version": "1.0", "status": "completed", "split": args.split,
        "subject_count": len(rows), "test_subject_count_read": len(rows) if args.split == "test" else 0,
        "direction_count": args.direction_count, "single_member": provenance,
        "output_root": str(output_root), "elapsed_seconds": time.perf_counter() - started,
        "all_finite": True, "subjects": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "inference_report.json").write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
