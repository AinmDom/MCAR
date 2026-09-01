"""Leakage-safe Q26 condition loading for FiLM-SIREN.

The public condition object contains only inference-time inputs.  Dense
reference spectra, MCA spectra and supervised residual targets never cross
this API boundary.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from mcar.fsp_ae_data import (
    SONICOM_DIRECTION_COUNT,
    SONICOM_Q26_COUNT,
    q26_source_indices,
)


def normalize_split(value: object) -> str:
    """Return the canonical train/val/test spelling for a scalar value."""
    scalar = np.asarray(value).item()
    text = scalar.decode() if isinstance(scalar, bytes) else str(scalar)
    normalized = text.strip().lower()
    if normalized == "validation":
        normalized = "val"
    if normalized not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split value {text!r}")
    return normalized


def verify_split_access(
    split_csv: Path,
    subject_id: int,
    *,
    allow_test: bool = False,
) -> tuple[str, str]:
    """Resolve ``subject_id`` through the frozen split and enforce test guard."""
    if int(subject_id) < 1:
        raise ValueError("subject_id must be positive")
    label = f"P{int(subject_id):04d}"
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("subject_id") == label
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected one split row for {label}, found {len(rows)}")
    split = normalize_split(rows[0].get("split", ""))
    if split == "test" and not allow_test:
        raise PermissionError(
            f"test subject {label} requires the explicit allow_test flag"
        )
    return label, split


@dataclass(frozen=True)
class SparseCondition:
    """Variable-size sparse binaural magnitude condition for one subject."""

    subject_label: str
    source_indices: np.ndarray
    binaural_magnitude_db: np.ndarray
    xyz: np.ndarray
    mask: np.ndarray
    frequency_hz: np.ndarray

    @property
    def direction_count(self) -> int:
        return int(self.source_indices.size)

    @property
    def frequency_count(self) -> int:
        return int(self.frequency_hz.size)

    def validate(self) -> None:
        q = self.direction_count
        f = self.frequency_count
        if self.source_indices.shape != (q,) or q < 1:
            raise ValueError("source_indices must be a non-empty vector")
        if len(set(self.source_indices.tolist())) != q:
            raise ValueError("source_indices must be unique")
        if np.any(self.source_indices < 0) or np.any(
            self.source_indices >= SONICOM_DIRECTION_COUNT
        ):
            raise ValueError("source_indices lie outside the reference grid")
        if self.binaural_magnitude_db.shape != (2, q, f):
            raise ValueError(f"binaural_magnitude_db must have shape [2,{q},{f}]")
        if self.xyz.shape != (q, 3):
            raise ValueError(f"xyz must have shape [{q},3]")
        if self.mask.shape != (q,) or self.mask.dtype != np.bool_:
            raise ValueError(f"mask must be bool with shape [{q}]")
        if not bool(np.any(self.mask)):
            raise ValueError("all-masked condition input is invalid")
        for name, values in (
            ("binaural_magnitude_db", self.binaural_magnitude_db),
            ("xyz", self.xyz),
            ("frequency_hz", self.frequency_hz),
        ):
            if not bool(np.all(np.isfinite(values))):
                raise ValueError(f"{name} must be finite")
        if f < 2 or not bool(np.all(np.diff(self.frequency_hz) > 0.0)):
            raise ValueError("frequency_hz must be strictly increasing")


@dataclass(frozen=True)
class Q26Condition(SparseCondition):
    """The original frozen 26-direction condition for one subject."""

    def validate(self) -> None:
        super().validate()
        if self.source_indices.shape != (SONICOM_Q26_COUNT,):
            raise ValueError(
                f"source_indices must have shape [{SONICOM_Q26_COUNT}]"
            )


def build_sparse_condition(
    dataset_root: Path,
    split_csv: Path,
    subject_id: int,
    source_indices: np.ndarray,
    *,
    allow_test: bool = False,
) -> SparseCondition:
    """Load only the requested measured directions from a subject HDF5."""
    label, csv_split = verify_split_access(
        split_csv,
        subject_id,
        allow_test=allow_test,
    )
    indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if indices.size < 1 or len(set(indices.tolist())) != indices.size:
        raise ValueError("Sparse source indices must be non-empty and unique")
    if not bool(np.all(np.diff(indices) > 0)):
        raise ValueError("Sparse source indices must be strictly increasing")
    if np.any(indices < 0) or np.any(indices >= SONICOM_DIRECTION_COUNT):
        raise ValueError("Sparse source indices lie outside the reference grid")

    path = dataset_root / "subjects" / label / "q26.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        hdf5_split = normalize_split(handle.attrs["split"])
        actual_subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
        if actual_subject_id != int(subject_id):
            raise ValueError(
                f"HDF5 identity mismatch for {path}: subject={actual_subject_id}"
            )
        if hdf5_split != csv_split:
            raise ValueError(
                f"Split mismatch for {path}: csv={csv_split!r}, "
                f"hdf5={hdf5_split!r}"
            )
        if hdf5_split == "test" and not allow_test:
            raise PermissionError(
                f"test HDF5 {path} requires the explicit allow_test flag"
            )
        reference = handle["reference_logmag_db"]
        directions = handle["direction_features"]
        frequency_dataset = handle["frequency_hz"]
        if reference.shape[:2] != (2, SONICOM_DIRECTION_COUNT):
            raise ValueError(f"Unexpected reference layout {reference.shape}")
        if directions.shape != (SONICOM_DIRECTION_COUNT, 6):
            raise ValueError(f"Unexpected direction layout {directions.shape}")
        magnitude = np.asarray(reference[:, indices, :], dtype=np.float32)
        xyz = np.asarray(directions[indices, 2:5], dtype=np.float32)
        frequency = np.asarray(frequency_dataset[:], dtype=np.float32).reshape(-1)

    condition = SparseCondition(
        subject_label=label,
        source_indices=indices.copy(),
        binaural_magnitude_db=magnitude,
        xyz=xyz,
        mask=np.ones(indices.size, dtype=bool),
        frequency_hz=frequency,
    )
    condition.validate()
    return condition


def build_q26_condition(
    dataset_root: Path,
    split_csv: Path,
    subject_id: int,
    q26_csv: Path,
    *,
    allow_test: bool = False,
    mask: np.ndarray | None = None,
) -> Q26Condition:
    """Load only the Q26 inference-time inputs for one subject."""
    label, csv_split = verify_split_access(
        split_csv,
        subject_id,
        allow_test=allow_test,
    )
    indices = q26_source_indices(q26_csv)
    if not bool(np.all(np.diff(indices) > 0)):
        raise ValueError(
            "Q26 source indices must be strictly increasing for frozen HDF5 access"
        )

    path = dataset_root / "subjects" / label / "q26.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        hdf5_split = normalize_split(handle.attrs["split"])
        actual_subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
        if actual_subject_id != int(subject_id):
            raise ValueError(
                f"HDF5 identity mismatch for {path}: subject={actual_subject_id}"
            )
        if hdf5_split != csv_split:
            raise ValueError(
                f"Split mismatch for {path}: csv={csv_split!r}, "
                f"hdf5={hdf5_split!r}"
            )
        if hdf5_split == "test" and not allow_test:
            raise PermissionError(
                f"test HDF5 {path} requires the explicit allow_test flag"
            )

        reference = handle["reference_logmag_db"]
        directions = handle["direction_features"]
        frequency_dataset = handle["frequency_hz"]
        sparse_dataset = handle["sparse_direction_indices_zero_based"]
        if reference.shape[:2] != (2, SONICOM_DIRECTION_COUNT):
            raise ValueError(f"Unexpected reference layout {reference.shape}")
        if directions.shape != (SONICOM_DIRECTION_COUNT, 6):
            raise ValueError(f"Unexpected direction layout {directions.shape}")
        if sparse_dataset.shape not in {
            (SONICOM_Q26_COUNT,),
            (SONICOM_Q26_COUNT, 1),
        }:
            raise ValueError(f"Unexpected sparse-index layout {sparse_dataset.shape}")
        stored_indices = np.asarray(sparse_dataset[:], dtype=np.int64).reshape(-1)
        if not np.array_equal(stored_indices, indices):
            raise ValueError(
                f"HDF5 sparse indices do not match frozen Q26 CSV for {path}"
            )

        magnitude = np.asarray(reference[:, indices, :], dtype=np.float32)
        xyz = np.asarray(directions[indices, 2:5], dtype=np.float32)
        frequency = np.asarray(frequency_dataset[:], dtype=np.float32).reshape(-1)

    valid_mask = (
        np.ones(indices.size, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    if valid_mask.shape != (indices.size,):
        raise ValueError(f"mask must have shape [{indices.size}]")
    condition = Q26Condition(
        subject_label=label,
        source_indices=indices.astype(np.int64, copy=True),
        binaural_magnitude_db=magnitude,
        xyz=xyz,
        mask=valid_mask.copy(),
        frequency_hz=frequency,
    )
    condition.validate()
    return condition


@dataclass(frozen=True)
class Q26MagnitudeNormalization:
    """Per-ear, per-frequency statistics computed from all train subjects."""

    mean_db: np.ndarray
    std_db: np.ndarray
    frequency_hz: np.ndarray
    training_subject_count: int

    @classmethod
    def from_json(cls, path: Path) -> "Q26MagnitudeNormalization":
        content = json.loads(path.read_text(encoding="utf-8"))
        instance = cls(
            mean_db=np.asarray(content["mean_db"], dtype=np.float32),
            std_db=np.asarray(content["std_population_db"], dtype=np.float32),
            frequency_hz=np.asarray(content["frequency_hz"], dtype=np.float32),
            training_subject_count=int(content["training_subject_count"]),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        if self.mean_db.ndim != 2 or self.mean_db.shape[0] != 2:
            raise ValueError("Q26 normalization mean must have shape [2,F]")
        if self.std_db.shape != self.mean_db.shape:
            raise ValueError("Q26 normalization std must match mean")
        if self.frequency_hz.shape != (self.mean_db.shape[1],):
            raise ValueError("Q26 normalization frequency vector is incompatible")
        if self.training_subject_count < 1:
            raise ValueError("training_subject_count must be positive")
        if not bool(np.all(np.isfinite(self.mean_db))):
            raise ValueError("Q26 normalization mean must be finite")
        if not bool(np.all(np.isfinite(self.std_db))) or bool(
            np.any(self.std_db <= 0.0)
        ):
            raise ValueError("Q26 normalization std must be finite and positive")

    def normalize(self, magnitude_db: np.ndarray) -> np.ndarray:
        values = np.asarray(magnitude_db, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 2:
            raise ValueError("Q26 magnitude must have shape [2,Q,F]")
        if values.shape[2] != self.mean_db.shape[1]:
            raise ValueError("Q26 magnitude frequency count does not match statistics")
        return (
            (values - self.mean_db[:, None, :])
            / self.std_db[:, None, :]
        ).astype(np.float32)
