"""Run the frozen Bounded-E25 ensemble at Q14, Q26, and Q50 on validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.paths import project_root
from mcar.predictors import BoundedMcarFilmCorrectionPredictor
from mcar.training.train_film_siren import file_sha256, split_subject_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "configs/experiments/"
            "sonicom_bounded_e25_input_direction_sensitivity_v1_manifest.json"
        ),
    )
    parser.add_argument("--directions-per-block", type=int, default=32)
    parser.add_argument("--subject-limit", type=int)
    return parser.parse_args()


def identity(payload: dict[str, object]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256"))
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Sensitivity manifest identity mismatch")
    return actual


def grid_indices(path: Path, count: int) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        indices = [
            int(row["source_index_zero_based"])
            for row in csv.DictReader(handle)
            if int(row["direction_count"]) == count
        ]
    result = np.asarray(indices, dtype=np.int64)
    if result.shape != (count,) or not bool(np.all(np.diff(result) > 0)):
        raise ValueError(f"Invalid frozen Q{count} grid")
    return result


def write_prediction(
    path: Path,
    prediction: np.ndarray,
    *,
    subject_id: int,
    subject_label: str,
    count: int,
    manifest_identity: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w") as handle:
        handle.create_dataset(
            "predicted_residual_db",
            data=prediction,
            chunks=(1, 64, prediction.shape[2]),
            compression="gzip",
            compression_opts=4,
        )
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["complete"] = 1
        handle.attrs["subject_id"] = subject_id
        handle.attrs["subject_label"] = subject_label
        handle.attrs["split"] = "val"
        handle.attrs["sparse_direction_count"] = count
        handle.attrs["tensor_layout"] = "ear,direction,frequency"
        handle.attrs["model_family"] = "BoundedMcarFilmCorrectionEnsemble"
        handle.attrs["manifest_identity_sha256"] = manifest_identity
    partial.replace(path)


def main() -> None:
    arguments = parse_arguments()
    if arguments.directions_per_block < 1:
        raise ValueError("directions-per-block must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Bounded-E25 sensitivity inference")

    root = project_root()
    manifest_path = arguments.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_identity = identity(manifest)
    if manifest["status"] != "frozen_before_inference":
        raise ValueError("Sensitivity manifest is not frozen before inference")
    config_path = root / manifest["experiment_config"]
    if file_sha256(config_path).upper() != manifest["experiment_config_sha256"]:
        raise ValueError("Experiment config hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dataset"]["split"] != "val" or config["inference"]["test_access_allowed"]:
        raise PermissionError("Sensitivity inference must remain validation-only")

    model_manifest_path = root / config["paper_main_model"]["manifest"]
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest["identity_sha256"] != config["paper_main_model"][
        "manifest_identity_sha256"
    ]:
        raise ValueError("Paper-main-model identity mismatch")
    weights = np.asarray(model_manifest["ensemble"]["weights"], dtype=np.float64)
    if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Expected the frozen three-member equal-weight ensemble")

    dataset_root = (root / config["dataset"]["root"]).resolve()
    split_csv = (root / config["dataset"]["split_file"]).resolve()
    grid_csv = (root / config["dataset"]["sparse_grid_file"]).resolve()
    q26_csv = (root / model_manifest["dataset"]["q26_csv"]).resolve()
    normalization = (root / config["inference"]["normalization"]).resolve()
    subjects = split_subject_paths(dataset_root, split_csv, "val")
    if len(subjects) != 44:
        raise ValueError("Expected exactly 44 validation subjects")
    if arguments.subject_limit is not None:
        subjects = subjects[: arguments.subject_limit]
    artifact_root = root / "artifacts/sparsity" / config["output_name"]
    prior_q26_root = root / (
        "artifacts/reconstruction/"
        "sonicom_bounded_mcar_film_correction_final_e25_ensemble_validation"
    )
    execution_counts = (26, 14, 50)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    q26_reproduction_max = 0.0

    for count in execution_counts:
        indices = grid_indices(grid_csv, count)
        predictors: list[BoundedMcarFilmCorrectionPredictor] = []
        for member in model_manifest["members"]:
            checkpoint = root / member["checkpoint"]
            if file_sha256(checkpoint).upper() != member["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for seed {member['seed']}")
            predictors.append(
                BoundedMcarFilmCorrectionPredictor(
                    checkpoint,
                    split_csv,
                    q26_csv,
                    normalization,
                    device=torch.device("cuda"),
                    directions_per_block=arguments.directions_per_block,
                    allow_test=False,
                    condition_source_indices=indices,
                    condition_dataset_root=dataset_root,
                )
            )
        for subject_index, (_, subject_label, source_path) in enumerate(subjects, 1):
            level_root = artifact_root / "subjects" / subject_label / f"q{count}"
            input_path = level_root / "model_input.h5"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            predictions = [predictor.predict_residual_db(input_path) for predictor in predictors]
            prediction = np.average(
                np.stack(predictions), axis=0, weights=weights
            ).astype(np.float32)
            if prediction.shape != (2, 793, 463) or not bool(
                np.all(np.isfinite(prediction))
            ):
                raise FloatingPointError(f"Invalid Q{count} prediction for {subject_label}")
            if count == 26:
                prior_path = prior_q26_root / "subjects" / subject_label / "prediction.h5"
                with h5py.File(prior_path, "r") as prior:
                    prior_prediction = np.asarray(
                        prior["predicted_residual_db"][:], dtype=np.float32
                    )
                q26_reproduction_max = max(
                    q26_reproduction_max,
                    float(np.max(np.abs(prediction - prior_prediction))),
                )
            write_prediction(
                level_root / "bounded_prediction.h5",
                prediction,
                subject_id=int(subject_label[1:]),
                subject_label=subject_label,
                count=count,
                manifest_identity=manifest_identity,
            )
            rows.append(
                {
                    "subject_label": subject_label,
                    "direction_count": count,
                    "shape": list(prediction.shape),
                    "finite": True,
                }
            )
            print(
                f"Bounded E25 sensitivity Q{count} "
                f"[{subject_index}/{len(subjects)}] {subject_label}",
                flush=True,
            )
        del predictors
        torch.cuda.empty_cache()
        if count == 26 and q26_reproduction_max > float(
            config["inference"]["q26_prediction_reproduction_tolerance_db"]
        ):
            raise ValueError(
                "Q26 reproduction failed: "
                f"max_abs={q26_reproduction_max:.9g} dB"
            )

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "val",
        "subject_count": len(subjects),
        "direction_counts": [14, 26, 50],
        "prediction_count": len(rows),
        "all_finite": all(bool(row["finite"]) for row in rows),
        "q26_reproduction_max_abs_error_db": q26_reproduction_max,
        "q26_reproduction_tolerance_db": config["inference"][
            "q26_prediction_reproduction_tolerance_db"
        ],
        "manifest": str(manifest_path),
        "manifest_identity_sha256": manifest_identity,
        "test_subject_count_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": rows,
    }
    (artifact_root / "inference_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
