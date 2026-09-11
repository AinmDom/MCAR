#!/usr/bin/env python3
"""Validate frozen ARI MCA/FSC feature HDF5 files without SONICOM assumptions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def validate(path: Path, expected_split: str | None) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        arrays = {name: handle[name][:] for name in (
            "mca_logmag_db", "reference_logmag_db", "correction_logmag_db",
            "target_residual_db", "direction_features", "frequency_hz",
            "sparse_direction_indices_zero_based", "interpolation_evaluation_mask",
        )}
        mca = arrays["mca_logmag_db"]
        reference = arrays["reference_logmag_db"]
        correction = arrays["correction_logmag_db"]
        residual = arrays["target_residual_db"]
        features = arrays["direction_features"]
        frequency = np.squeeze(arrays["frequency_hz"])
        sparse = np.squeeze(arrays["sparse_direction_indices_zero_based"]).astype(np.int64)
        interpolation = np.squeeze(arrays["interpolation_evaluation_mask"]).astype(bool)
        split = handle.attrs["split"]
        split = split.decode() if isinstance(split, bytes) else str(split)
        if expected_split is not None and split != expected_split:
            raise ValueError(f"{path}: split {split!r} != expected {expected_split!r}")
        if int(np.asarray(handle.attrs["test_subjects_read"]).item()) != 0:
            raise ValueError(f"{path}: nonzero test_subjects_read")
        if (int(np.asarray(handle.attrs["complete"]).item()) != 1 or
                int(np.asarray(handle.attrs["strict_ild_metadata"]).item()) != 1):
            raise ValueError(f"{path}: incomplete strict-ILD feature file")
        if mca.shape != (2, 1550, 463) or any(item.shape != mca.shape for item in (reference, correction, residual)):
            raise ValueError(f"{path}: invalid spectral shapes")
        if features.shape != (1550, 6) or frequency.shape != (463,):
            raise ValueError(f"{path}: invalid direction/frequency shapes")
        if sparse.shape != (26,) or len(np.unique(sparse)) != 26:
            raise ValueError(f"{path}: invalid Q26 indices")
        expected_mask = np.ones(1550, dtype=bool); expected_mask[sparse] = False
        if not np.array_equal(interpolation, expected_mask):
            raise ValueError(f"{path}: interpolation mask is not complement of Q26")
        weights = features[:, 5]
        if not np.all(weights[sparse] == 0.0) or np.any(weights[interpolation] <= 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{path}: frozen ARI non-observation Voronoi weights invalid")
        strict = handle["strict_ild"]
        phase = strict["mca_selected_phase_rad"][:]
        outside_real = strict["mca_outside_real"][:]
        outside_imag = strict["mca_outside_imag"][:]
        ild = np.squeeze(strict["reference_ild_db"][:])
        selected = np.squeeze(strict["selected_bin_indices_zero_based"][:]).astype(np.int64)
        outside = np.squeeze(strict["outside_bin_indices_zero_based"][:]).astype(np.int64)
        if phase.shape != mca.shape or outside_real.shape != (2, 1550, 50) or outside_imag.shape != outside_real.shape or ild.shape != (1550,):
            raise ValueError(f"{path}: strict ILD tensor shape invalid")
        if selected.size != 463 or outside.size != 50 or not np.array_equal(np.sort(np.r_[selected, outside]), np.arange(513)):
            raise ValueError(f"{path}: strict ILD FFT bins invalid")
        values = (mca, reference, correction, residual, features, frequency, phase, outside_real, outside_imag, ild)
        if not all(np.isfinite(item).all() for item in values):
            raise ValueError(f"{path}: non-finite feature content")
        identity = float(np.max(np.abs((reference - mca) - residual)))
        if identity > 2e-5:
            raise ValueError(f"{path}: residual identity error {identity}")
        return {"file": str(path), "split": split, "subject_label": str(handle.attrs["subject_label"]), "identity_max_db": identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-split", choices=("train", "val"))
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    results = [validate(path, args.expected_split) for path in args.files]
    print(f"validated_files={len(results)} max_identity_db={max(row['identity_max_db'] for row in results):.9g}")


if __name__ == "__main__":
    main()
