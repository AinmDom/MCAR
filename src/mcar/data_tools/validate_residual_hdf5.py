"""Validate one or more MCA residual-learning HDF5 files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "mca_logmag_db",
    "reference_logmag_db",
    "correction_logmag_db",
    "target_residual_db",
    "direction_features",
    "frequency_hz",
)


def scalar_attribute(handle: h5py.File, name: str) -> object:
    return np.asarray(handle.attrs[name]).item()


def validate_file(path: Path) -> dict[str, float | int | str]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"{path}: missing datasets: {missing}")

        mca = handle["mca_logmag_db"][:]
        reference = handle["reference_logmag_db"][:]
        correction = handle["correction_logmag_db"][:]
        residual = handle["target_residual_db"][:]
        directions = handle["direction_features"][:]
        frequency = np.squeeze(handle["frequency_hz"][:])

        if mca.shape != reference.shape or mca.shape != correction.shape:
            raise ValueError(f"{path}: feature tensor shapes differ")
        if mca.shape != residual.shape:
            raise ValueError(f"{path}: target shape {residual.shape} != {mca.shape}")
        if len(mca.shape) != 3 or mca.shape[0] != 2 or mca.shape[1] != 900:
            raise ValueError(
                f"{path}: expected Python layout [2, 900, F], got {mca.shape}"
            )
        if directions.shape != (900, 6):
            raise ValueError(f"{path}: expected directions [900, 6], got {directions.shape}")
        if frequency.ndim != 1 or frequency.size != mca.shape[2]:
            raise ValueError(f"{path}: frequency vector does not match tensor")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError(f"{path}: frequencies are not strictly increasing")
        if frequency[0] < 50 or frequency[-1] > 20_000:
            raise ValueError(f"{path}: frequency range is outside 50 Hz-20 kHz")

        for name, values in (
            ("mca", mca),
            ("reference", reference),
            ("correction", correction),
            ("residual", residual),
            ("directions", directions),
            ("frequency", frequency),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: {name} contains non-finite values")

        reconstruction_error = float(
            np.max(np.abs((reference - mca) - residual))
        )
        if reconstruction_error > 2e-5:
            raise ValueError(
                f"{path}: residual identity error {reconstruction_error:.3g} dB"
            )
        if int(scalar_attribute(handle, "complete")) != 1:
            raise ValueError(f"{path}: file is not marked complete")

        return {
            "file": str(path),
            "subject_id": int(scalar_attribute(handle, "subject_id")),
            "sparse_order": int(scalar_attribute(handle, "sparse_order")),
            "split": handle.attrs["split"].decode()
            if isinstance(handle.attrs["split"], bytes)
            else str(handle.attrs["split"]),
            "frequency_count": int(frequency.size),
            "sample_count": int(residual.size),
            "residual_mean_db": float(np.mean(residual)),
            "residual_std_db": float(np.std(residual)),
            "residual_abs_mean_db": float(np.mean(np.abs(residual))),
            "residual_identity_max_error_db": reconstruction_error,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Validate every file but print only one aggregate summary.",
    )
    parser.add_argument("files", nargs="+", type=Path)
    arguments = parser.parse_args()

    results = []
    for file_path in arguments.files:
        result = validate_file(file_path)
        results.append(result)
        if not arguments.summary_only:
            print(
                "{file}: subject={subject_id}, N={sparse_order}, split={split}, "
                "shape_samples={sample_count}, frequencies={frequency_count}, "
                "mean={residual_mean_db:.6f} dB, std={residual_std_db:.6f} dB, "
                "|residual|={residual_abs_mean_db:.6f} dB, identity_error="
                "{residual_identity_max_error_db:.3g} dB".format(**result)
            )

    if arguments.summary_only:
        split_counts = Counter(str(result["split"]) for result in results)
        total_samples = sum(int(result["sample_count"]) for result in results)
        maximum_identity_error = max(
            float(result["residual_identity_max_error_db"])
            for result in results
        )
        mean_absolute_residual = float(
            np.mean(
                [float(result["residual_abs_mean_db"]) for result in results]
            )
        )
        print(
            f"validated_files={len(results)}, total_samples={total_samples}, "
            f"split_counts={dict(sorted(split_counts.items()))}, "
            f"mean_subject_abs_residual_db={mean_absolute_residual:.6f}, "
            f"max_identity_error_db={maximum_identity_error:.3g}"
        )


if __name__ == "__main__":
    main()
