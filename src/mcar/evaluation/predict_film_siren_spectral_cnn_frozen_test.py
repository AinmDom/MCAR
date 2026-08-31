"""Run the single authorized Hybrid E190 frozen-test inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mcar.evaluation.predict_film_siren_ensemble import decode_attribute, split_subject_paths
from mcar.frozen_test import atomic_update_registry, load_verified_manifest, load_verified_registry
from mcar.paths import project_root
from mcar.predictors import FilmSirenSpectralCNNPredictor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--directions-per-block", type=int, default=64)
    return parser.parse_args()


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
    if split != "test":
        raise ValueError(f"Expected test source, found {split!r}")
    temporary = output_path.with_suffix(".h5.partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary, "w") as destination:
        destination.create_dataset(
            "predicted_residual_db",
            data=prediction,
            chunks=(1, min(64, prediction.shape[1]), prediction.shape[2]),
            compression="gzip",
            compression_opts=4,
        )
        destination.attrs.update(
            {
                "schema_version": "1.0",
                "complete": 1,
                "subject_id": subject_id,
                "subject_label": subject_label,
                "split": split,
                "tensor_layout": "ear,direction,frequency",
                "model_family": "FilmSirenSpectralCNNEnsemble",
                "manifest_identity_sha256": manifest["identity_sha256"],
            }
        )
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
    if not arguments.allow_test:
        raise PermissionError("Frozen SONICOM test requires explicit --allow-test")
    if arguments.directions_per_block < 1:
        raise ValueError("directions-per-block must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen inference")

    root = project_root()
    manifest_path = arguments.manifest.resolve()
    manifest = load_verified_manifest(manifest_path, root)
    if manifest["model_version"] != "film-siren-spectral-cnn-final-e190-ensemble-test-v1":
        raise ValueError("Unexpected frozen candidate identity")
    registry_path = arguments.registry.resolve()
    registry = load_verified_registry(registry_path, manifest)
    output_root = root / "artifacts" / "reconstruction" / arguments.run_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")

    atomic_update_registry(registry_path, registry, "started", run_name=arguments.run_name)
    try:
        dataset = manifest["dataset"]
        subject_rows = split_subject_paths(
            root / dataset["root"], root / dataset["split_csv"], "test"
        )
        if len(subject_rows) != 44:
            raise ValueError(f"Expected 44 test subjects, found {len(subject_rows)}")
        predictors = [
            FilmSirenSpectralCNNPredictor(
                root / member["checkpoint"],
                root / dataset["split_csv"],
                root / dataset["q26_csv"],
                root / dataset["q26_normalization"],
                device=torch.device("cuda"),
                directions_per_block=arguments.directions_per_block,
                allow_test=True,
            )
            for member in manifest["members"]
        ]
        weights = np.asarray(manifest["ensemble"]["weights"], dtype=np.float64)
        if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Expected three weights summing to one")

        started = time.perf_counter()
        rows = []
        for index, source_path in enumerate(subject_rows, start=1):
            with h5py.File(source_path, "r") as source:
                subject_label = decode_attribute(source.attrs["subject_label"])
            prediction = np.average(
                np.stack([p.predict_residual_db(source_path) for p in predictors]),
                axis=0,
                weights=weights,
            ).astype(np.float32)
            if not np.all(np.isfinite(prediction)):
                raise FloatingPointError(subject_label)
            rows.append(
                write_prediction(
                    source_path,
                    output_root / "subjects" / subject_label / "prediction.h5",
                    prediction,
                    manifest,
                )
            )
            print(f"predicted [{index}/44] {subject_label}", flush=True)

        report = {
            "schema_version": "1.0",
            "status": "completed",
            "split": "test",
            "subject_count": 44,
            "test_subject_count_read": 44,
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
        atomic_update_registry(
            registry_path,
            registry,
            "completed",
            run_name=arguments.run_name,
            inference_report=str(report_path),
            test_subject_count_read=44,
        )
    except Exception as error:
        atomic_update_registry(
            registry_path,
            registry,
            "failed",
            run_name=arguments.run_name,
            failure=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
