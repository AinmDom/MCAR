"""Predict a hash-locked three-member FiLM-SIREN residual ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mcar.frozen_test import (
    atomic_update_registry,
    load_verified_manifest,
    load_verified_registry,
)
from mcar.paths import project_root
from mcar.predictors import FilmSirenPredictor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--subject-limit", type=int)
    return parser.parse_args()


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def split_subject_paths(dataset_root: Path, split_csv: Path, split: str) -> list[Path]:
    """Resolve only the requested split without scanning unrelated HDF5 files."""
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = [
            row["subject_id"]
            for row in csv.DictReader(handle)
            if row.get("split") == split
        ]
    if len(labels) != 44 or len(set(labels)) != 44:
        raise ValueError(f"Expected 44 unique {split} subjects, found {len(labels)}")
    paths = [dataset_root / "subjects" / label / "q26.h5" for label in labels]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def write_prediction(
    source_path: Path,
    output_path: Path,
    prediction: np.ndarray,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    with h5py.File(source_path, "r") as source:
        subject_id = int(np.asarray(source.attrs["subject_id"]).item())
        subject_label = decode_attribute(source.attrs["subject_label"])
        split = decode_attribute(source.attrs["split"])
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
        destination.attrs["model_family"] = "FilmSirenResidualEnsemble"
        destination.attrs["manifest_identity_sha256"] = manifest["identity_sha256"]
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
    root = project_root()
    manifest_path = arguments.manifest.resolve()
    manifest = load_verified_manifest(manifest_path, root)
    is_test = arguments.split == "test"
    registry: dict[str, Any] | None = None
    registry_path: Path | None = None
    if is_test:
        if not arguments.allow_test:
            raise PermissionError("Frozen SONICOM test requires explicit --allow-test")
        if arguments.registry is None:
            raise ValueError("Frozen SONICOM test requires --registry")
        registry_path = arguments.registry.resolve()
        registry = load_verified_registry(registry_path, manifest)
        atomic_update_registry(
            registry_path,
            registry,
            "started",
            manifest=str(manifest_path),
            run_name=arguments.run_name,
        )
    elif arguments.allow_test or arguments.registry is not None:
        raise ValueError("--allow-test and --registry are only valid with --split test")
    if arguments.directions_per_block < 1:
        raise ValueError("directions-per-block must be positive")
    if arguments.subject_limit is not None and arguments.subject_limit < 1:
        raise ValueError("subject-limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen FiLM-SIREN inference")

    try:
        dataset_root = (root / manifest["dataset"]["root"]).resolve()
        split_csv = root / manifest["dataset"]["split_csv"]
        subject_files = split_subject_paths(dataset_root, split_csv, arguments.split)
        if arguments.subject_limit is not None:
            subject_files = subject_files[: arguments.subject_limit]
        q26_csv = root / manifest["dataset"]["q26_csv"]
        q26_normalization = root / manifest["dataset"]["q26_normalization"]
        predictors = [
            FilmSirenPredictor(
                root / member["checkpoint"],
                split_csv,
                q26_csv,
                q26_normalization,
                device=torch.device("cuda"),
                directions_per_block=arguments.directions_per_block,
                allow_test=is_test,
            )
            for member in manifest["members"]
        ]
        weights = np.asarray(manifest["ensemble"]["weights"], dtype=np.float64)
        if weights.shape != (len(predictors),) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Ensemble weights must match members and sum to one")
        output_root = root / "artifacts" / "reconstruction" / arguments.run_name
        if output_root.exists():
            raise FileExistsError(f"Refusing to overwrite {output_root}")

        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for index, source_path in enumerate(subject_files, start=1):
            member_predictions = [
                predictor.predict_residual_db(source_path) for predictor in predictors
            ]
            prediction = np.average(
                np.stack(member_predictions, axis=0),
                axis=0,
                weights=weights,
            ).astype(np.float32)
            if not bool(np.all(np.isfinite(prediction))):
                raise FloatingPointError(f"Non-finite ensemble output for {source_path}")
            with h5py.File(source_path, "r") as source:
                subject_label = decode_attribute(source.attrs["subject_label"])
            output_path = output_root / "subjects" / subject_label / "prediction.h5"
            rows.append(write_prediction(source_path, output_path, prediction, manifest))
            print(f"predicted [{index}/{len(subject_files)}] {subject_label}", flush=True)
        report = {
            "schema_version": "1.0",
            "status": "completed",
            "split": arguments.split,
            "subject_count": len(subject_files),
            "test_subject_count_read": len(subject_files) if is_test else 0,
            "manifest": str(manifest_path),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "ensemble": manifest["ensemble"],
            "run_name": arguments.run_name,
            "output_root": str(output_root.resolve()),
            "elapsed_seconds": time.perf_counter() - started,
            "subjects": rows,
        }
        report_path = output_root / "inference_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if is_test:
            assert registry is not None and registry_path is not None
            atomic_update_registry(
                registry_path,
                registry,
                "completed",
                manifest=str(manifest_path),
                run_name=arguments.run_name,
                inference_report=str(report_path.resolve()),
                test_subject_count_read=len(subject_files),
            )
        print(json.dumps(report, indent=2))
    except Exception as error:
        if is_test and registry is not None and registry_path is not None:
            atomic_update_registry(
                registry_path,
                registry,
                "failed",
                manifest=str(manifest_path),
                run_name=arguments.run_name,
                failure=f"{type(error).__name__}: {error}",
            )
        raise


if __name__ == "__main__":
    main()
