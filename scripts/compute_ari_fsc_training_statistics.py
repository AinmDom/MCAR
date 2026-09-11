#!/usr/bin/env python3
"""Compute ARI FSC normalization statistics from frozen train features only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def mean_std(files: list[Path], dataset: str) -> tuple[float, float, int]:
    count = 0; total = 0.0; square = 0.0
    for path in files:
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle[dataset][:], dtype=np.float64)
        count += values.size; total += float(values.sum()); square += float(np.square(values).sum())
    mean = total / count
    return mean, max(0.0, square / count - mean * mean) ** 0.5, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q26-output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input.glob("nh*.h5"))
    if len(files) != 130:
        raise ValueError(f"expected frozen train=130 features, found {len(files)}")
    frequency = None; directions = None; identifiers: list[str] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            split = handle.attrs["split"]
            split = split.decode() if isinstance(split, bytes) else str(split)
            if split != "train" or int(np.asarray(handle.attrs["test_subjects_read"]).item()) != 0:
                raise ValueError(f"split/test-access violation: {path}")
            current_frequency = np.squeeze(handle["frequency_hz"][:]).astype(float)
            current_directions = handle["direction_features"][:].astype(float)
            if frequency is None:
                frequency, directions = current_frequency, current_directions
            elif not np.array_equal(frequency, current_frequency) or not np.array_equal(directions, current_directions):
                raise ValueError(f"grid mismatch: {path}")
            identifiers.append(str(handle.attrs["subject_label"]))
    assert frequency is not None and directions is not None
    tensors = {}
    for name in ("mca_logmag_db", "correction_logmag_db", "target_residual_db"):
        mean, std, count = mean_std(files, name)
        tensors[name] = {"mean": mean, "std_population": std, "count": count}
    result = {
        "schema_version": "1.0", "source": "ARI frozen train split only",
        "training_subject_count": len(files), "training_subject_labels": identifiers,
        "test_subjects_read": 0, "all_finite": True,
        "per_tensor": tensors,
        "frequency": {"count": int(frequency.size), "minimum_hz": float(frequency.min()), "maximum_hz": float(frequency.max()), "log10_mean": float(np.log10(frequency).mean()), "log10_std_population": float(np.log10(frequency).std())},
        "direction_features": {"feature_order": ["azimuth_deg", "elevation_deg", "x", "y", "z", "evaluation_voronoi_weight"], "shape": list(directions.shape)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    q_sum = np.zeros((2, 463), dtype=np.float64); q_square = np.zeros((2, 463), dtype=np.float64); q_count = 0
    q_indices = None
    for path in files:
        with h5py.File(path, "r") as handle:
            indices = np.squeeze(handle["sparse_direction_indices_zero_based"][:]).astype(np.int64)
            values = np.asarray(handle["reference_logmag_db"][:], dtype=np.float64)[:, indices, :]
            if q_indices is None:
                q_indices = indices
            elif not np.array_equal(q_indices, indices):
                raise ValueError(f"Q26 drift: {path}")
            q_sum += values.sum(axis=1); q_square += np.square(values).sum(axis=1); q_count += values.shape[1]
    q_mean = q_sum / q_count; q_std = np.sqrt(np.maximum(0.0, q_square / q_count - q_mean * q_mean))
    q_result = {"schema_version": "1.0", "source": "ARI Q26 reference_logmag_db from frozen train split only", "training_subject_count": len(files), "q26_direction_count": 26, "frequency_hz": frequency.tolist(), "mean_db": q_mean.tolist(), "std_population_db": q_std.tolist(), "test_subjects_read": 0}
    if not np.isfinite(q_std).all() or (q_std <= 0).any():
        raise ValueError("invalid ARI Q26 normalization")
    args.q26_output.parent.mkdir(parents=True, exist_ok=True)
    args.q26_output.write_text(json.dumps(q_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training_subjects": len(files), "target_mean": tensors["target_residual_db"]["mean"], "target_std": tensors["target_residual_db"]["std_population"], "test_subjects_read": 0}))


if __name__ == "__main__":
    main()
