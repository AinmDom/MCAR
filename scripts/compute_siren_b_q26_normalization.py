"""Compute Stage B Q26 input statistics from all 262 train subjects only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root
from mcar.q26_condition import build_q26_condition


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def train_subject_ids(split_csv: Path) -> list[int]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["subject_id"] for row in rows if row["split"].lower() == "train"]
    if len(labels) != 262 or len(set(labels)) != 262:
        raise ValueError(f"Expected 262 unique train subjects, found {len(labels)}")
    if any(not label.startswith("P") or not label[1:].isdigit() for label in labels):
        raise ValueError("Invalid subject label in split CSV")
    return [int(label[1:]) for label in labels]


def compute_statistics(
    dataset_root: Path,
    split_csv: Path,
    q26_csv: Path,
) -> dict[str, object]:
    subject_ids = train_subject_ids(split_csv)
    total: np.ndarray | None = None
    total_square: np.ndarray | None = None
    frequency_hz: np.ndarray | None = None
    value_count = 0
    for subject_id in subject_ids:
        condition = build_q26_condition(
            dataset_root,
            split_csv,
            subject_id,
            q26_csv,
        )
        values = condition.binaural_magnitude_db.astype(np.float64)
        if total is None:
            total = np.zeros((2, values.shape[2]), dtype=np.float64)
            total_square = np.zeros_like(total)
            frequency_hz = condition.frequency_hz.astype(np.float64)
        elif not np.array_equal(frequency_hz, condition.frequency_hz):
            raise ValueError(f"Frequency grid changed at {condition.subject_label}")
        total += np.sum(values, axis=1)
        total_square += np.sum(np.square(values), axis=1)
        value_count += values.shape[1]
    if total is None or total_square is None or frequency_hz is None:
        raise AssertionError("No train condition was accumulated")
    mean = total / value_count
    variance = np.maximum(total_square / value_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0.0) or not np.all(np.isfinite(std)):
        raise ValueError("Computed Q26 standard deviation is invalid")
    return {
        "schema_version": "1.0",
        "source": "Q26 reference_logmag_db from complete frozen train split",
        "training_subject_count": len(subject_ids),
        "training_subject_ids": subject_ids,
        "q26_direction_count": int(value_count // len(subject_ids)),
        "per_ear_frequency_sample_count": value_count,
        "frequency_hz": frequency_hz.tolist(),
        "mean_db": mean.tolist(),
        "std_population_db": std.tolist(),
        "split_csv": "configs/data/sonicom_subject_split_v1.csv",
        "split_csv_sha256": file_sha256(split_csv),
        "q26_csv": "configs/data/sonicom_sparse_grid_q26_v1.csv",
        "q26_csv_sha256": file_sha256(q26_csv),
        "test_subjects_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    root = project_root()
    dataset_root = root / "data" / "processed" / "sonicom_residual_q26_v1"
    split_csv = root / "configs" / "data" / "sonicom_subject_split_v1.csv"
    q26_csv = root / "configs" / "data" / "sonicom_sparse_grid_q26_v1.csv"
    output = arguments.output or (
        root / "configs" / "data" / "siren_b_q26_normalization_v1.json"
    )
    statistics = compute_statistics(dataset_root, split_csv, q26_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "training_subject_count": statistics["training_subject_count"],
                "test_subjects_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
