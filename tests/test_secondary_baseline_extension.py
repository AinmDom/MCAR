from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_film_secondary_baseline_extension_validation.py"
)
SPEC = importlib.util.spec_from_file_location("secondary_baseline_extension", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_ranf_prediction_preserves_direction_ear_and_selected_bins(tmp_path: Path) -> None:
    path = tmp_path / "prediction.sofa"
    directions = np.zeros((793, 6), dtype=np.float32)
    directions[:, 0] = np.linspace(-179.0, 179.0, 793)
    directions[:, 1] = np.linspace(-80.0, 80.0, 793)
    hrir = np.zeros((793, 2, 256), dtype=np.float32)
    hrir[:, 0, 0] = 1.0
    hrir[:, 1, 1] = 0.5
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Data.IR", data=hrir)
        handle.create_dataset("Data.SamplingRate", data=np.asarray([44100.0]))
        handle.create_dataset("SourcePosition", data=directions[:, :3])
    selected = np.asarray([2, 7, 31], dtype=np.int64)
    actual_db, actual_hrir = MODULE.load_ranf_prediction(
        path, selected, directions, 44100.0
    )
    expected = np.fft.fft(hrir.astype(np.float64), n=1024, axis=-1)[..., selected]
    expected_db = np.transpose(
        20.0 * np.log10(np.maximum(np.abs(expected), 1e-10)), (1, 0, 2)
    )
    assert actual_db.shape == (2, 793, 3)
    assert np.allclose(actual_db, expected_db)
    assert np.array_equal(actual_hrir, hrir)


def test_load_fspae_prediction_applies_one_bin_offset(tmp_path: Path) -> None:
    path = tmp_path / "prediction.h5"
    magnitude = np.arange(793 * 2 * 512, dtype=np.float32).reshape(793, 2, 512)
    hrir = np.zeros((793, 2, 256), dtype=np.float32)
    frequency = np.arange(1, 513, dtype=np.float32) * (44100.0 / 1024.0)
    selected = np.asarray([2, 7, 31], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.attrs["split"] = "val"
        handle.attrs["subject_id"] = "P0001"
        handle.create_dataset("predicted_magnitude_db", data=magnitude)
        handle.create_dataset("predicted_hrir", data=hrir)
        handle.create_dataset("frequency_hz", data=frequency)
    actual_db, actual_hrir = MODULE.load_fspae_prediction(
        path, selected, frequency[selected - 1], "P0001"
    )
    expected = np.transpose(magnitude[..., selected - 1], (1, 0, 2))
    assert actual_db.shape == (2, 793, 3)
    assert np.array_equal(actual_db, expected)
    assert np.array_equal(actual_hrir, hrir)
