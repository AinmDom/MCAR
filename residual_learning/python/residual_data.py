"""HDF5 block sampling for the MCA residual-learning data set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class Normalization:
    mca_mean: float
    mca_std: float
    correction_mean: float
    correction_std: float
    target_mean: float
    target_std: float
    log_frequency_mean: float
    log_frequency_std: float

    @classmethod
    def from_json(cls, path: Path) -> "Normalization":
        content = json.loads(path.read_text(encoding="utf-8"))
        tensors = content["per_tensor"]
        return cls(
            mca_mean=float(tensors["mca_logmag_db"]["mean"]),
            mca_std=float(tensors["mca_logmag_db"]["std_population"]),
            correction_mean=float(tensors["correction_logmag_db"]["mean"]),
            correction_std=float(tensors["correction_logmag_db"]["std_population"]),
            target_mean=float(tensors["target_residual_db"]["mean"]),
            target_std=float(tensors["target_residual_db"]["std_population"]),
            log_frequency_mean=float(content["frequency"]["log10_mean"]),
            log_frequency_std=float(content["frequency"]["log10_std_population"]),
        )


def _decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def list_hdf5_files(
    dataset_root: Path, split: str | None = None, subject_ids: Sequence[int] | None = None
) -> list[Path]:
    requested_subjects = set(subject_ids) if subject_ids is not None else None
    files: list[Path] = []
    for path in sorted((dataset_root / "subjects").glob("pp*/n*.h5")):
        with h5py.File(path, "r") as handle:
            current_split = _decode_attribute(handle.attrs["split"])
            subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
            if split is not None and current_split != split:
                continue
            if requested_subjects is not None and subject_id not in requested_subjects:
                continue
            files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No HDF5 files match split={split!r}, subject_ids={subject_ids!r} "
            f"under {dataset_root}"
        )
    return files


class ResidualBlockSampler:
    """Sample Cartesian direction-frequency blocks from independent HDF5 files.

    Each returned batch contains all combinations of `directions_per_batch`
    directions and `frequencies_per_batch` frequencies for one randomly chosen
    subject and ear. This keeps HDF5 reads contiguous enough for efficient I/O.
    """

    def __init__(
        self,
        files: Sequence[Path],
        normalization: Normalization,
        directions_per_batch: int = 64,
        frequencies_per_batch: int = 128,
        seed: int = 20260723,
    ) -> None:
        self.files = list(files)
        self.normalization = normalization
        self.directions_per_batch = directions_per_batch
        self.frequencies_per_batch = frequencies_per_batch
        self.rng = np.random.default_rng(seed)
        self._check_layout()

    @property
    def batch_size(self) -> int:
        return self.directions_per_batch * self.frequencies_per_batch

    def _check_layout(self) -> None:
        with h5py.File(self.files[0], "r") as handle:
            shape = handle["mca_logmag_db"].shape
            direction_shape = handle["direction_features"].shape
            if len(shape) != 3 or shape[0] != 2 or shape[1] != 900:
                raise ValueError(f"Unexpected spectral shape {shape} in {self.files[0]}")
            if direction_shape != (900, 6):
                raise ValueError(
                    f"Unexpected direction feature shape {direction_shape} in {self.files[0]}"
                )
            if self.directions_per_batch > shape[1]:
                raise ValueError("directions_per_batch exceeds available directions")
            if self.frequencies_per_batch > shape[2]:
                raise ValueError("frequencies_per_batch exceeds available frequencies")

    def sample_batch(self) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
        path = self.files[int(self.rng.integers(len(self.files)))]
        with h5py.File(path, "r") as handle:
            ear_index = int(self.rng.integers(2))
            direction_indices = np.sort(
                self.rng.choice(900, size=self.directions_per_batch, replace=False)
            )
            frequency_count = handle["mca_logmag_db"].shape[2]
            frequency_indices = np.sort(
                self.rng.choice(
                    frequency_count, size=self.frequencies_per_batch, replace=False
                )
            )
            mca = handle["mca_logmag_db"][ear_index, direction_indices, :][
                :, frequency_indices
            ]
            correction = handle["correction_logmag_db"][ear_index, direction_indices, :][
                :, frequency_indices
            ]
            target = handle["target_residual_db"][ear_index, direction_indices, :][
                :, frequency_indices
            ]
            direction_features = handle["direction_features"][direction_indices, :]
            frequency_hz = np.squeeze(handle["frequency_hz"][:])[frequency_indices]
            metadata: dict[str, int | str] = {
                "subject_id": int(np.asarray(handle.attrs["subject_id"]).item()),
                "sparse_order": int(np.asarray(handle.attrs["sparse_order"]).item()),
                "split": _decode_attribute(handle.attrs["split"]),
                "ear_index": ear_index,
            }

        features, target_normalized = build_features(
            mca, correction, target, direction_features, frequency_hz, ear_index, self.normalization
        )
        return features, target_normalized, metadata


def build_features(
    mca_db: np.ndarray,
    correction_db: np.ndarray,
    target_db: np.ndarray,
    direction_features: np.ndarray,
    frequency_hz: np.ndarray,
    ear_index: int,
    normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct normalized model inputs for a [direction, frequency] block."""
    mca_db = np.asarray(mca_db, dtype=np.float32)
    correction_db = np.asarray(correction_db, dtype=np.float32)
    target_db = np.asarray(target_db, dtype=np.float32)
    direction_features = np.asarray(direction_features, dtype=np.float32)
    frequency_hz = np.asarray(frequency_hz, dtype=np.float32)
    if mca_db.shape != correction_db.shape or mca_db.shape != target_db.shape:
        raise ValueError("MCA, correction, and target blocks must have the same shape")
    if mca_db.shape != (direction_features.shape[0], frequency_hz.size):
        raise ValueError("Direction/frequency metadata does not match spectral block")

    direction_count, frequency_count = mca_db.shape
    direction_xyz = np.repeat(direction_features[:, 2:5], frequency_count, axis=0)
    normalized_frequency = (
        np.log10(frequency_hz) - normalization.log_frequency_mean
    ) / normalization.log_frequency_std
    normalized_frequency = np.tile(normalized_frequency, direction_count)[:, None]
    ear_feature = np.full(
        (direction_count * frequency_count, 1), -1.0 if ear_index == 0 else 1.0, dtype=np.float32
    )
    features = np.concatenate(
        (
            ((mca_db - normalization.mca_mean) / normalization.mca_std).reshape(-1, 1),
            ((correction_db - normalization.correction_mean) / normalization.correction_std).reshape(-1, 1),
            direction_xyz,
            normalized_frequency.astype(np.float32),
            ear_feature,
        ),
        axis=1,
        dtype=np.float32,
    )
    target_normalized = (
        (target_db.reshape(-1, 1) - normalization.target_mean) / normalization.target_std
    ).astype(np.float32)
    return features, target_normalized


def iterate_file_blocks(
    path: Path,
    normalization: Normalization,
    directions_per_block: int = 64,
    frequencies_per_block: int = 128,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield every sample in one HDF5 file without loading the full tensor."""
    with h5py.File(path, "r") as handle:
        frequency_hz = np.squeeze(handle["frequency_hz"][:])
        directions = handle["direction_features"][:]
        frequency_count = frequency_hz.size
        for ear_index in range(2):
            for direction_start in range(0, 900, directions_per_block):
                direction_slice = slice(
                    direction_start, min(900, direction_start + directions_per_block)
                )
                for frequency_start in range(0, frequency_count, frequencies_per_block):
                    frequency_slice = slice(
                        frequency_start,
                        min(frequency_count, frequency_start + frequencies_per_block),
                    )
                    mca = handle["mca_logmag_db"][ear_index, direction_slice, frequency_slice]
                    correction = handle["correction_logmag_db"][
                        ear_index, direction_slice, frequency_slice
                    ]
                    target = handle["target_residual_db"][
                        ear_index, direction_slice, frequency_slice
                    ]
                    yield build_features(
                        mca,
                        correction,
                        target,
                        directions[direction_slice, :],
                        frequency_hz[frequency_slice],
                        ear_index,
                        normalization,
                    )
