"""SONICOM data access and cache format for the FSP-AE baseline."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from mcar.fsp_ae_signal import (
    estimate_itd_seconds,
    magnitude_db_from_hrir,
    positive_frequency_hz,
)


SONICOM_DIRECTION_COUNT = 793
SONICOM_HRIR_LENGTH = 256
SONICOM_SAMPLING_RATE_HZ = 44_100.0
SONICOM_SOURCE_RADIUS_M = 1.5
SONICOM_NFFT = 1024
SONICOM_FREQUENCY_COUNT = SONICOM_NFFT // 2
SONICOM_Q26_COUNT = 26


@dataclass(frozen=True)
class FSPAESubjectData:
    subject_id: str
    split: str
    hrtf_magnitude_db: torch.Tensor
    itd_seconds: torch.Tensor
    frequency_hz: torch.Tensor
    source_positions_cartesian_m: torch.Tensor
    sampling_rate_hz: float


def _normalize_split(split: str) -> str:
    normalized = split.lower()
    if normalized == "validation":
        normalized = "val"
    if normalized not in {"train", "val", "test"}:
        raise ValueError("split must be train, val/validation, or test")
    return normalized


def subject_ids_for_split(
    split_csv: Path,
    split: str,
    allow_test: bool = False,
) -> list[str]:
    """Read frozen subject IDs, refusing test before touching any SOFA file."""
    normalized = _normalize_split(split)
    if normalized == "test" and not allow_test:
        raise PermissionError(
            "test subject access requires the explicit allow_test flag"
        )
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    subject_ids = [
        row["subject_id"]
        for row in rows
        if _normalize_split(row["split"]) == normalized
    ]
    if not subject_ids:
        raise ValueError(f"no subjects found for split {normalized!r}")
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError(f"duplicate subject IDs in {split_csv}")
    return subject_ids


def q26_source_indices(q26_csv: Path) -> np.ndarray:
    with q26_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = np.asarray(
        [int(row["source_index_zero_based"]) for row in rows], dtype=np.int64
    )
    if indices.shape != (SONICOM_Q26_COUNT,) or len(set(indices.tolist())) != 26:
        raise ValueError("frozen Q26 configuration must contain 26 unique indices")
    if np.any(indices < 0) or np.any(indices >= SONICOM_DIRECTION_COUNT):
        raise ValueError("Q26 source index is outside the SONICOM reference grid")
    return indices


def resolve_sofa_path(sofa_root: Path, subject_id: str) -> Path:
    path = sofa_root / f"{subject_id}_FreeFieldCompMinPhase_44kHz.sofa"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def spherical_degrees_to_cartesian_m(
    source_positions: np.ndarray,
) -> torch.Tensor:
    positions = np.asarray(source_positions, dtype=np.float64)
    if positions.shape != (SONICOM_DIRECTION_COUNT, 3):
        raise ValueError(
            f"expected SourcePosition [793,3], found {positions.shape}"
        )
    azimuth = np.deg2rad(positions[:, 0])
    elevation = np.deg2rad(positions[:, 1])
    radius = positions[:, 2]
    cartesian = np.stack(
        (
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ),
        axis=-1,
    )
    return torch.from_numpy(cartesian.astype(np.float32))


def load_sonicom_sofa(
    sofa_path: Path,
    subject_id: str,
    split: str,
    nfft: int = SONICOM_NFFT,
    itd_direction_batch_size: int = 64,
) -> FSPAESubjectData:
    """Load and transform one retained-ITD SONICOM SOFA file."""
    normalized_split = _normalize_split(split)
    with h5py.File(sofa_path, "r") as handle:
        convention = handle.attrs.get("SOFAConventions", b"")
        if isinstance(convention, bytes):
            convention = convention.decode()
        if convention != "SimpleFreeFieldHRIR":
            raise ValueError(f"unexpected SOFA convention {convention!r}")
        hrir = np.asarray(handle["Data.IR"], dtype=np.float32)
        sampling_rate = float(np.asarray(handle["Data.SamplingRate"]).item())
        source_positions = np.asarray(handle["SourcePosition"], dtype=np.float64)
    if hrir.shape != (
        SONICOM_DIRECTION_COUNT,
        2,
        SONICOM_HRIR_LENGTH,
    ):
        raise ValueError(f"unexpected SONICOM HRIR shape {hrir.shape}")
    if not np.isclose(sampling_rate, SONICOM_SAMPLING_RATE_HZ):
        raise ValueError(f"unexpected sampling rate {sampling_rate}")
    if not np.allclose(source_positions[:, 2], SONICOM_SOURCE_RADIUS_M):
        raise ValueError("SONICOM source radius differs from frozen 1.5 m")
    hrir_tensor = torch.from_numpy(hrir)
    magnitude_db = magnitude_db_from_hrir(hrir_tensor, nfft=nfft)
    itd_seconds = estimate_itd_seconds(
        hrir_tensor,
        sampling_rate,
        direction_batch_size=itd_direction_batch_size,
    )
    frequency_hz = positive_frequency_hz(sampling_rate, nfft=nfft)
    if magnitude_db.shape != (
        SONICOM_DIRECTION_COUNT,
        2,
        nfft // 2,
    ):
        raise AssertionError("magnitude conversion returned an unexpected shape")
    if not all(
        torch.all(torch.isfinite(values))
        for values in (magnitude_db, itd_seconds, frequency_hz)
    ):
        raise ValueError(f"non-finite FSP-AE input in {sofa_path}")
    return FSPAESubjectData(
        subject_id=subject_id,
        split=normalized_split,
        hrtf_magnitude_db=magnitude_db,
        itd_seconds=itd_seconds,
        frequency_hz=frequency_hz,
        source_positions_cartesian_m=spherical_degrees_to_cartesian_m(
            source_positions
        ),
        sampling_rate_hz=sampling_rate,
    )


def cache_path(cache_root: Path, split: str, subject_id: str) -> Path:
    return cache_root / "subjects" / _normalize_split(split) / f"{subject_id}.h5"


def write_subject_cache(data: FSPAESubjectData, path: Path) -> None:
    """Atomically write one reproducible subject cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with h5py.File(partial, "w") as handle:
            handle.create_dataset(
                "hrtf_magnitude_db",
                data=data.hrtf_magnitude_db.numpy(),
                chunks=(64, 2, data.frequency_hz.numel()),
                compression="gzip",
                compression_opts=4,
            )
            handle.create_dataset(
                "itd_seconds", data=data.itd_seconds.numpy()
            )
            handle.create_dataset(
                "frequency_hz", data=data.frequency_hz.numpy()
            )
            handle.create_dataset(
                "source_positions_cartesian_m",
                data=data.source_positions_cartesian_m.numpy(),
            )
            handle.attrs["subject_id"] = data.subject_id
            handle.attrs["split"] = data.split
            handle.attrs["sampling_rate_hz"] = data.sampling_rate_hz
            handle.attrs["nfft"] = 2 * data.frequency_hz.numel()
            handle.attrs["magnitude_top_db"] = 80.0
            handle.attrs["itd_definition"] = (
                "1.6 kHz low-pass, 384 kHz resampling, +/-1 ms cross-correlation"
            )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def read_subject_cache(path: Path) -> FSPAESubjectData:
    with h5py.File(path, "r") as handle:
        subject_id = str(handle.attrs["subject_id"])
        split = _normalize_split(str(handle.attrs["split"]))
        sampling_rate = float(handle.attrs["sampling_rate_hz"])
        data = FSPAESubjectData(
            subject_id=subject_id,
            split=split,
            hrtf_magnitude_db=torch.from_numpy(
                np.asarray(handle["hrtf_magnitude_db"], dtype=np.float32)
            ),
            itd_seconds=torch.from_numpy(
                np.asarray(handle["itd_seconds"], dtype=np.float32)
            ),
            frequency_hz=torch.from_numpy(
                np.asarray(handle["frequency_hz"], dtype=np.float32)
            ),
            source_positions_cartesian_m=torch.from_numpy(
                np.asarray(
                    handle["source_positions_cartesian_m"], dtype=np.float32
                )
            ),
            sampling_rate_hz=sampling_rate,
        )
    return data


def list_cache_files(
    cache_root: Path,
    split: str,
    allow_test: bool = False,
) -> list[Path]:
    normalized = _normalize_split(split)
    if normalized == "test" and not allow_test:
        raise PermissionError("test cache access requires allow_test")
    files = sorted((cache_root / "subjects" / normalized).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no {normalized} FSP-AE caches under {cache_root}")
    return files


def calculate_normalization(files: Sequence[Path]) -> dict[str, float | int]:
    """Calculate population statistics without loading every subject at once."""
    if not files:
        raise ValueError("at least one cache file is required")
    totals = {"hrtf_mag": 0.0, "itd": 0.0}
    squares = {"hrtf_mag": 0.0, "itd": 0.0}
    counts = {"hrtf_mag": 0, "itd": 0}
    for path in files:
        with h5py.File(path, "r") as handle:
            for key, dataset_name in (
                ("hrtf_mag", "hrtf_magnitude_db"),
                ("itd", "itd_seconds"),
            ):
                values = np.asarray(handle[dataset_name], dtype=np.float64)
                totals[key] += float(np.sum(values))
                squares[key] += float(np.sum(np.square(values)))
                counts[key] += int(values.size)
    result: dict[str, float | int] = {"subject_count": len(files)}
    for key in ("hrtf_mag", "itd"):
        mean = totals[key] / counts[key]
        variance = max(squares[key] / counts[key] - mean * mean, 0.0)
        result[f"{key}_mean"] = mean
        result[f"{key}_std_population"] = float(np.sqrt(variance))
        result[f"{key}_sample_count"] = counts[key]
    return result


def write_normalization(
    path: Path,
    statistics: dict[str, float | int],
    expected_subject_count: int,
) -> None:
    payload = dict(statistics)
    payload["expected_subject_count"] = expected_subject_count
    payload["complete"] = statistics["subject_count"] == expected_subject_count
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_normalization_to_model(
    model,
    statistics: dict[str, float | int],
    dataset_name: str = "sonicom",
) -> None:
    model.set_stats(
        float(statistics["hrtf_mag_mean"]),
        float(statistics["hrtf_mag_std_population"]),
        dataset_name,
        "hrtf_mag",
    )
    model.set_stats(
        float(statistics["itd_mean"]),
        float(statistics["itd_std_population"]),
        dataset_name,
        "itd",
    )
