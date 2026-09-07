"""Explicitly authorized test inference for a frozen FSC E190 ensemble.

The validation-only ensemble entry point is intentionally not broadened.  This
separate command requires ``--allow-test`` and uses an isolated test residual
dataset exported with ``allowTest=true``.
"""
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
from mcar.training.train_film_siren import file_sha256


def manifest_identity(payload: dict[str, Any]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256")).upper()
    actual = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Manifest identity mismatch")
    return actual


def decode(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def test_subject_paths(split_csv: Path, dataset_root: Path, count: int) -> list[tuple[int, str, Path]]:
    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if len(rows) != 44:
        raise ValueError(f"Expected 44 test subjects, found {len(rows)}")
    output = []
    for row in rows:
        label = row["subject_id"]
        path = dataset_root / "subjects" / label / f"q{count}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append((int(label[1:]), label, path))
    return output


def write_prediction_test(
    source_path: Path,
    output_path: Path,
    prediction: np.ndarray,
    manifest_path: Path,
    identity: str,
) -> dict[str, Any]:
    with h5py.File(source_path, "r") as source:
        subject_id = int(np.asarray(source.attrs["subject_id"]).item())
        subject_label = decode(source.attrs["subject_label"])
        split = decode(source.attrs["split"])
    if split != "test":
        raise PermissionError(f"Expected test source, got {split}: {source_path}")
    if prediction.shape != (2, 793, 463) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError(f"Invalid test prediction: {subject_label}")
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary, "w") as destination:
        destination.create_dataset(
            "predicted_residual_db",
            data=prediction.astype(np.float32),
            chunks=(1, min(64, prediction.shape[1]), prediction.shape[2]),
            compression="gzip",
            compression_opts=4,
        )
        destination.attrs.update(
            schema_version="1.0",
            complete=1,
            subject_id=subject_id,
            subject_label=subject_label,
            split="test",
            tensor_layout="ear,direction,frequency",
            model_family="FilmSirenSpectralCNNEnsemble",
            checkpoint=str(manifest_path),
            checkpoint_sha256=identity,
        )
    temporary.replace(output_path)
    return {"subject_id": subject_id, "subject_label": subject_label, "shape": list(prediction.shape), "finite": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("test_dataset_root", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--directions-per-block", type=int, default=64)
    args = parser.parse_args()
    if not args.allow_test:
        raise PermissionError("Explicit --allow-test is required for test inference")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FSC test inference")
    root = project_root()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest_identity(manifest)
    if manifest.get("status") != "frozen" or manifest.get("e_final") != 190:
        raise ValueError("Frozen E190 manifest required")
    split_csv = (root / manifest["dataset"]["split_csv"]).resolve()
    q26_csv = (root / manifest["dataset"]["q26_csv"]).resolve()
    normalization = (root / manifest["dataset"]["q26_normalization"]).resolve()
    test_root = args.test_dataset_root.resolve()
    count = int(manifest["dataset"]["observation_count"])
    indices = sparse_source_indices(q26_csv, count)
    subject_rows = test_subject_paths(split_csv, test_root, count)
    predictors = []
    for member in manifest["members"]:
        checkpoint = (root / member["checkpoint"]).resolve()
        config = (root / member["config"]).resolve()
        if file_sha256(checkpoint).upper() != member["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        if file_sha256(config).upper() != member["config_sha256"]:
            raise ValueError(f"Config hash mismatch: {config}")
        predictors.append(
            FilmSirenSpectralCNNPredictor(
                checkpoint,
                split_csv,
                q26_csv,
                normalization,
                device=torch.device("cuda"),
                directions_per_block=args.directions_per_block,
                allow_test=True,
                condition_source_indices=indices,
                condition_dataset_root=test_root,
                condition_filename=f"q{count}.h5",
            )
        )
    output_root = root / "artifacts/reconstruction" / args.run_name
    if output_root.exists():
        raise FileExistsError(output_root)
    weights = np.asarray(manifest["ensemble"]["weights"], dtype=np.float64)
    if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Invalid three-seed ensemble weights")
    started = time.perf_counter()
    rows = []
    for index, (_, label, source) in enumerate(subject_rows, 1):
        members = [predictor.predict_residual_db(source) for predictor in predictors]
        prediction = np.average(np.stack(members, axis=0), axis=0, weights=weights)
        rows.append(write_prediction_test(source, output_root / "subjects" / label / "prediction.h5", prediction, manifest_path, identity))
        print(f"test predicted [{index}/44] {label}", flush=True)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "test",
        "subject_count": len(rows),
        "test_subject_count_read": len(rows),
        "manifest": str(manifest_path),
        "manifest_identity_sha256": identity,
        "dataset_root": str(test_root),
        "direction_count": count,
        "ensemble": manifest["ensemble"],
        "elapsed_seconds": time.perf_counter() - started,
        "all_finite": True,
        "subjects": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "inference_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
