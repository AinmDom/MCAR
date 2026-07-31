"""Compute streaming normalization statistics from training subjects only."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class RunningStatistics:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    absolute_total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        self.count += values64.size
        self.total += float(np.sum(values64))
        self.total_square += float(np.sum(values64 * values64))
        self.absolute_total += float(np.sum(np.abs(values64)))
        self.minimum = min(self.minimum, float(np.min(values64)))
        self.maximum = max(self.maximum, float(np.max(values64)))

    def as_dict(self) -> dict[str, float | int]:
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std_population": variance**0.5,
            "mean_absolute": self.absolute_total / self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to DATASET_ROOT/training_statistics.json.",
    )
    arguments = parser.parse_args()

    output_path = arguments.output or (
        arguments.dataset_root / "training_statistics.json"
    )
    files = sorted((arguments.dataset_root / "subjects").glob("*/*.h5"))
    if not files:
        raise FileNotFoundError(f"No subject HDF5 files under {arguments.dataset_root}")

    statistics = {
        "mca_logmag_db": RunningStatistics(),
        "correction_logmag_db": RunningStatistics(),
        "target_residual_db": RunningStatistics(),
    }
    train_subjects: list[int] = []
    frequency_hz: np.ndarray | None = None
    direction_features: np.ndarray | None = None
    direction_feature_order: str | None = None

    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            if decode_attribute(handle.attrs["split"]) != "train":
                continue
            train_subjects.append(int(np.asarray(handle.attrs["subject_id"]).item()))
            for dataset_name, running in statistics.items():
                running.update(handle[dataset_name][:])
            current_frequency = np.squeeze(handle["frequency_hz"][:]).astype(np.float64)
            current_direction = handle["direction_features"][:].astype(np.float64)
            current_feature_order = decode_attribute(
                handle.attrs["direction_feature_order"]
            )
            if frequency_hz is None:
                frequency_hz = current_frequency
                direction_features = current_direction
                direction_feature_order = current_feature_order
            else:
                if not np.array_equal(frequency_hz, current_frequency):
                    raise ValueError(f"Frequency grid differs in {file_path}")
                if not np.array_equal(direction_features, current_direction):
                    raise ValueError(f"Direction grid differs in {file_path}")
                if direction_feature_order != current_feature_order:
                    raise ValueError(
                        f"Direction feature order differs in {file_path}"
                    )

    if not train_subjects:
        raise ValueError("No training subjects were found")
    assert (
        frequency_hz is not None
        and direction_features is not None
        and direction_feature_order is not None
    )

    log_frequency = np.log10(frequency_hz)
    result = {
        "schema_version": "1.0",
        "source": "train split only",
        "training_subject_count": len(train_subjects),
        "training_subject_ids": sorted(train_subjects),
        "per_tensor": {
            name: running.as_dict() for name, running in statistics.items()
        },
        "frequency": {
            "count": int(frequency_hz.size),
            "minimum_hz": float(frequency_hz.min()),
            "maximum_hz": float(frequency_hz.max()),
            "log10_mean": float(log_frequency.mean()),
            "log10_std_population": float(log_frequency.std()),
        },
        "direction_features": {
            "feature_order": direction_feature_order.split(","),
            "mean": direction_features.mean(axis=0).tolist(),
            "std_population": direction_features.std(axis=0).tolist(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)

    target = result["per_tensor"]["target_residual_db"]
    print(
        f"training_subjects={len(train_subjects)}, "
        f"training_samples={target['count']}, "
        f"target_mean={target['mean']:.6f} dB, "
        f"target_std={target['std_population']:.6f} dB, "
        f"target_abs_mean={target['mean_absolute']:.6f} dB"
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
