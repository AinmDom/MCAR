"""Create the frozen validation or test residual ensemble for v3.5.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from mcar.paths import project_root


ROOT = project_root()
CANDIDATE_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v351b_band_ild005_e40"
    / "best.pt"
)
CANDIDATE_WEIGHT = 0.7


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def reconstruction_roots(split: str) -> tuple[Path, Path, Path]:
    if split == "val":
        names = (
            "sonicom_q26_validation_v321_cnn_reinit_joint_unfreeze_e10",
            "sonicom_q26_validation_v351b_best_total_e23",
            "sonicom_q26_validation_v351_previous30_b70",
        )
    else:
        names = (
            "sonicom_q26_test_v351_previous_joint",
            "sonicom_q26_test_v351b_best_total_e23",
            "sonicom_q26_test_v351_previous30_b70",
        )
    return (
        *(ROOT / "artifacts" / "reconstruction" / name for name in names),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    arguments = parse_arguments()
    if arguments.split == "test" and not arguments.allow_test:
        raise ValueError("The test split requires explicit --allow-test")
    previous_root, candidate_root, output_root = reconstruction_roots(
        arguments.split
    )
    previous_files = sorted(previous_root.glob("subjects/*/prediction.h5"))
    candidate_files = sorted(candidate_root.glob("subjects/*/prediction.h5"))
    if len(previous_files) != 44 or len(candidate_files) != 44:
        raise RuntimeError(
            f"Expected 44+44 {arguments.split} predictions, found "
            f"{len(previous_files)}+{len(candidate_files)}"
        )
    previous_subjects = [path.parent.name for path in previous_files]
    candidate_subjects = [path.parent.name for path in candidate_files]
    if previous_subjects != candidate_subjects:
        raise RuntimeError("Prediction subject lists do not match")

    for previous_path, candidate_path in zip(previous_files, candidate_files):
        subject = previous_path.parent.name
        output_path = output_root / "subjects" / subject / "prediction.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".h5.partial")
        temporary_path.unlink(missing_ok=True)
        with h5py.File(previous_path, "r") as previous_handle, h5py.File(
            candidate_path, "r"
        ) as candidate_handle:
            previous = np.asarray(
                previous_handle["predicted_residual_db"][:], dtype=np.float32
            )
            candidate = np.asarray(
                candidate_handle["predicted_residual_db"][:], dtype=np.float32
            )
            if previous.shape != candidate.shape:
                raise RuntimeError(f"Prediction shape mismatch for {subject}")
            fused = previous + np.float32(CANDIDATE_WEIGHT) * (candidate - previous)
            with h5py.File(temporary_path, "w") as destination:
                destination.create_dataset(
                    "predicted_residual_db",
                    data=fused,
                    chunks=(1, min(64, fused.shape[1]), fused.shape[2]),
                    compression="gzip",
                    compression_opts=4,
                )
                for name, value in previous_handle.attrs.items():
                    destination.attrs[name] = value
                destination.attrs["model_family"] = "ResidualOutputEnsemble"
                destination.attrs["model_version"] = "v3.5.1"
                destination.attrs["model_name"] = "SONICOM Q26 MCAR v3.5.1"
                destination.attrs["checkpoint"] = "output-level ensemble"
                destination.attrs["previous_prediction"] = str(previous_path.resolve())
                destination.attrs["candidate_prediction"] = str(candidate_path.resolve())
                destination.attrs["previous_weight"] = 1.0 - CANDIDATE_WEIGHT
                destination.attrs["candidate_weight"] = CANDIDATE_WEIGHT
                destination.attrs["selection_split"] = "val"
                destination.attrs["inference_split"] = arguments.split
                destination.attrs["test_subject_count_read"] = (
                    1 if arguments.split == "test" else 0
                )
        temporary_path.replace(output_path)

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "model_version": "v3.5.1",
        "formula": "0.3 * previous_joint + 0.7 * v351b_best_total_epoch23",
        "candidate_checkpoint": str(CANDIDATE_CHECKPOINT.relative_to(ROOT)),
        "candidate_checkpoint_sha256": sha256(CANDIDATE_CHECKPOINT),
        "split": arguments.split,
        "validation_subject_count": 44 if arguments.split == "val" else 0,
        "prediction_subject_count": 44,
        "output_root": str(output_root.relative_to(ROOT)),
        "test_subject_count_read": 44 if arguments.split == "test" else 0,
    }
    (output_root / "inference_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
