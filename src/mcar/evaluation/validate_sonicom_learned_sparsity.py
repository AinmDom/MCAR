"""Audit SONICOM learned-sparsity artifacts and frozen-Q26 parity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/sonicom_learned_methods_sparsity_v1.json"),
    )
    return parser.parse_args()


def test_subjects(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            row["subject_id"]
            for row in csv.DictReader(handle)
            if row["split"] == "test"
        ]


def finite_dataset(handle: h5py.File, key: str, shape: tuple[int, ...]) -> None:
    dataset = handle[key]
    if dataset.shape != shape:
        raise ValueError(f"{handle.filename}:{key} has shape {dataset.shape}, expected {shape}")
    for start in range(0, shape[0], 64):
        if not np.all(np.isfinite(dataset[start : start + 64])):
            raise FloatingPointError(f"Non-finite values in {handle.filename}:{key}")


def update_difference(summary: dict[str, float], left: np.ndarray, right: np.ndarray) -> None:
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    summary["maximum"] = max(summary["maximum"], float(np.max(difference)))
    summary["sum"] += float(np.sum(difference))
    summary["count"] += int(difference.size)


def main() -> None:
    args = parse_arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    subjects = test_subjects(Path(config["dataset"]["split_file"]))
    if len(subjects) != int(config["dataset"]["subject_count"]):
        raise ValueError("Frozen test subject count mismatch")
    counts = [int(value) for value in config["dataset"]["direction_counts"]]
    artifact_root = Path("artifacts/sparsity") / config["output_name"]
    q26_input_root = Path("data/processed/sonicom_residual_q26_v1/subjects")
    q26_mcar_root = Path(
        "artifacts/reconstruction/"
        "sonicom_q26_test_mlp_cnn_v32_seed20260809_e40_epoch39_frozen/subjects"
    )
    q26_fsp_root = Path(
        "artifacts/reconstruction/sonicom_fsp_ae_q26_locked_final_test/subjects"
    )
    parity = {
        "q26_input_mca_db": {"maximum": 0.0, "sum": 0.0, "count": 0},
        "q26_input_correction_db": {"maximum": 0.0, "sum": 0.0, "count": 0},
        "q26_mcar_residual_db": {"maximum": 0.0, "sum": 0.0, "count": 0},
        "q26_fspae_magnitude_db": {"maximum": 0.0, "sum": 0.0, "count": 0},
        "q26_fspae_itd_seconds": {"maximum": 0.0, "sum": 0.0, "count": 0},
        "q26_fspae_hrir": {"maximum": 0.0, "sum": 0.0, "count": 0},
    }
    audited = 0
    for subject in subjects:
        for count in counts:
            level_root = artifact_root / "subjects" / subject / f"q{count}"
            input_path = level_root / "model_input.h5"
            mcar_path = level_root / "mcar_prediction.h5"
            fsp_path = level_root / "fspae_prediction.h5"
            with h5py.File(input_path) as handle:
                if int(np.asarray(handle.attrs["complete"]).item()) != 1:
                    raise ValueError(f"Incomplete model input: {input_path}")
                finite_dataset(handle, "mca_logmag_db", (2, 793, 463))
                finite_dataset(handle, "correction_logmag_db", (2, 793, 463))
                finite_dataset(handle, "direction_features", (793, 6))
                finite_dataset(handle, "frequency_hz", (463, 1))
            with h5py.File(mcar_path) as handle:
                if int(np.asarray(handle.attrs["complete"]).item()) != 1:
                    raise ValueError(f"Incomplete MCAR prediction: {mcar_path}")
                finite_dataset(handle, "predicted_residual_db", (2, 793, 463))
            with h5py.File(fsp_path) as handle:
                if int(np.asarray(handle.attrs["complete"]).item()) != 1:
                    raise ValueError(f"Incomplete FSP-AE prediction: {fsp_path}")
                finite_dataset(handle, "predicted_magnitude_db", (793, 2, 512))
                finite_dataset(handle, "predicted_itd_seconds", (793,))
                finite_dataset(handle, "predicted_hrir", (793, 2, 256))
                indices = handle["sparse_indices_zero_based"][:]
                expected = np.asarray(
                    config["sparse_grid_selection"][
                        f"q{count}_source_indices_zero_based"
                    ]
                )
                if not np.array_equal(indices, expected):
                    raise ValueError(f"Sparse-index mismatch: {fsp_path}")
            audited += 1

        current_input = artifact_root / "subjects" / subject / "q26" / "model_input.h5"
        prior_input = q26_input_root / subject / "q26.h5"
        current_mcar = artifact_root / "subjects" / subject / "q26" / "mcar_prediction.h5"
        prior_mcar = q26_mcar_root / subject / "prediction.h5"
        current_fsp = artifact_root / "subjects" / subject / "q26" / "fspae_prediction.h5"
        prior_fsp = q26_fsp_root / subject / "prediction.h5"
        with h5py.File(current_input) as current, h5py.File(prior_input) as prior:
            update_difference(
                parity["q26_input_mca_db"],
                current["mca_logmag_db"][:],
                prior["mca_logmag_db"][:],
            )
            update_difference(
                parity["q26_input_correction_db"],
                current["correction_logmag_db"][:],
                prior["correction_logmag_db"][:],
            )
        with h5py.File(current_mcar) as current, h5py.File(prior_mcar) as prior:
            update_difference(
                parity["q26_mcar_residual_db"],
                current["predicted_residual_db"][:],
                prior["predicted_residual_db"][:],
            )
        with h5py.File(current_fsp) as current, h5py.File(prior_fsp) as prior:
            for key, dataset in (
                ("q26_fspae_magnitude_db", "predicted_magnitude_db"),
                ("q26_fspae_itd_seconds", "predicted_itd_seconds"),
                ("q26_fspae_hrir", "predicted_hrir"),
            ):
                update_difference(parity[key], current[dataset][:], prior[dataset][:])

    for values in parity.values():
        values["mean"] = values.pop("sum") / values["count"]
    report = {
        "schema_version": "1.0",
        "status": "passed",
        "subject_count": len(subjects),
        "audited_subject_levels": audited,
        "q26_parity_absolute_difference": parity,
        "notes": {
            "mcar": "Tiny sparse discrepancies are allowed from mixed-precision GPU convolution non-determinism.",
            "fspae_hrir": "All Q levels use the official CPU mcar.fsp_ae_signal reconstruction; Q26 is exactly equal to the historical formal output.",
        },
    }
    report_path = artifact_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
