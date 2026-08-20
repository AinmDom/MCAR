"""Leakage and schema tests for the Stage B Q26 condition interface."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from mcar.fsp_ae_data import q26_source_indices
from mcar.paths import project_root
from mcar.q26_condition import Q26Condition, build_q26_condition


SCRATCH = Path("artifacts") / "_q26_condition_test_tmp"
Q26_CSV = project_root() / "configs" / "data" / "sonicom_sparse_grid_q26_v1.csv"
REAL_DATASET_ROOT = (
    project_root() / "data" / "processed" / "sonicom_residual_q26_v1"
)


@pytest.fixture()
def synthetic_dataset() -> tuple[Path, Path]:
    shutil.rmtree(SCRATCH, ignore_errors=True)
    dataset_root = SCRATCH / "dataset"
    split_csv = SCRATCH / "split.csv"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject_id", "split"))
        writer.writeheader()
        writer.writerows(
            (
                {"subject_id": "P9001", "split": "train"},
                {"subject_id": "P9002", "split": "val"},
                {"subject_id": "P9003", "split": "test"},
            )
        )
    indices = q26_source_indices(Q26_CSV)
    frequency = np.arange(463, dtype=np.float32).reshape(463, 1) + 1.0
    directions = np.zeros((793, 6), dtype=np.float32)
    directions[:, 2] = np.linspace(-1.0, 1.0, 793, dtype=np.float32)
    for subject_id, split in ((9001, "train"), (9002, "val"), (9003, "test")):
        label = f"P{subject_id:04d}"
        path = dataset_root / "subjects" / label / "q26.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        reference = np.arange(2 * 793 * 463, dtype=np.float32).reshape(2, 793, 463)
        reference = reference + float(subject_id)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("reference_logmag_db", data=reference)
            handle.create_dataset("target_residual_db", data=-reference)
            handle.create_dataset("direction_features", data=directions)
            handle.create_dataset("frequency_hz", data=frequency)
            handle.create_dataset(
                "sparse_direction_indices_zero_based",
                data=indices.reshape(-1, 1).astype(np.int32),
            )
            handle.attrs["subject_id"] = np.asarray([subject_id], dtype=np.int32)
            handle.attrs["subject_label"] = label
            handle.attrs["split"] = split
    try:
        yield dataset_root, split_csv
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def test_q26_index_and_sparse_magnitude_correctness(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    condition = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    indices = q26_source_indices(Q26_CSV)
    with h5py.File(dataset_root / "subjects" / "P9001" / "q26.h5", "r") as h5:
        expected = np.asarray(h5["reference_logmag_db"][:, indices, :])
    np.testing.assert_array_equal(condition.source_indices, indices)
    np.testing.assert_allclose(condition.binaural_magnitude_db, expected)
    assert condition.binaural_magnitude_db.shape == (2, 26, 463)
    assert condition.xyz.shape == (26, 3)


def test_non_q26_mutation_invariance(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    before = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    q26_set = set(before.source_indices.tolist())
    non_q26 = np.asarray([i for i in range(793) if i not in q26_set])
    path = dataset_root / "subjects" / "P9001" / "q26.h5"
    with h5py.File(path, "r+") as handle:
        values = handle["reference_logmag_db"][:]
        values[:, non_q26, :] = np.nan
        handle["reference_logmag_db"][:] = values
    after = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    np.testing.assert_array_equal(after.source_indices, before.source_indices)
    np.testing.assert_allclose(after.binaural_magnitude_db, before.binaural_magnitude_db)
    np.testing.assert_allclose(after.xyz, before.xyz)


def test_target_mutation_and_removal_invariance(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    before = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    path = dataset_root / "subjects" / "P9001" / "q26.h5"
    with h5py.File(path, "r+") as handle:
        handle["target_residual_db"][:] = np.nan
    mutated = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    with h5py.File(path, "r+") as handle:
        del handle["target_residual_db"]
    removed = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    for candidate in (mutated, removed):
        np.testing.assert_array_equal(candidate.source_indices, before.source_indices)
        np.testing.assert_allclose(
            candidate.binaural_magnitude_db,
            before.binaural_magnitude_db,
        )
        np.testing.assert_allclose(candidate.xyz, before.xyz)


def test_public_api_exposes_no_supervised_or_dense_data(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    condition = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    assert set(condition.__dataclass_fields__) == {
        "subject_label",
        "source_indices",
        "binaural_magnitude_db",
        "xyz",
        "mask",
        "frequency_hz",
    }
    assert not hasattr(condition, "target_residual_db")
    assert not hasattr(condition, "reference_logmag_db")
    assert not hasattr(condition, "mca_logmag_db")


def test_q26_data_permutation_consistency(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    base = build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
    permutation = np.random.default_rng(7).permutation(26)
    permuted = Q26Condition(
        subject_label=base.subject_label,
        source_indices=base.source_indices[permutation],
        binaural_magnitude_db=base.binaural_magnitude_db[:, permutation, :],
        xyz=base.xyz[permutation],
        mask=base.mask[permutation],
        frequency_hz=base.frequency_hz,
    )
    permuted.validate()
    np.testing.assert_array_equal(
        permuted.binaural_magnitude_db,
        base.binaural_magnitude_db[:, permutation, :],
    )


def test_q26_mask_preservation_and_all_masked_rejection(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    mask = np.ones(26, dtype=bool)
    mask[[0, 3]] = False
    condition = build_q26_condition(
        dataset_root,
        split_csv,
        9001,
        Q26_CSV,
        mask=mask,
    )
    np.testing.assert_array_equal(condition.mask, mask)
    with pytest.raises(ValueError, match="all-masked"):
        build_q26_condition(
            dataset_root,
            split_csv,
            9001,
            Q26_CSV,
            mask=np.zeros(26, dtype=bool),
        )


def test_synthetic_test_split_requires_explicit_authorization(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    with pytest.raises(PermissionError, match="allow_test"):
        build_q26_condition(dataset_root, split_csv, 9003, Q26_CSV)
    condition = build_q26_condition(
        dataset_root,
        split_csv,
        9003,
        Q26_CSV,
        allow_test=True,
    )
    assert condition.subject_label == "P9003"


def test_validation_split_is_allowed(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    assert build_q26_condition(
        dataset_root,
        split_csv,
        9002,
        Q26_CSV,
    ).subject_label == "P9002"


def test_csv_hdf5_split_mismatch_is_rejected(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    path = dataset_root / "subjects" / "P9001" / "q26.h5"
    with h5py.File(path, "r+") as handle:
        handle.attrs["split"] = "val"
    with pytest.raises(ValueError, match="Split mismatch"):
        build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)


def test_hdf5_sparse_indices_match_frozen_csv() -> None:
    """The committed train artifact and frozen Q26 configuration must agree."""
    path = REAL_DATASET_ROOT / "subjects" / "P0289" / "q26.h5"
    with h5py.File(path, "r") as handle:
        stored = np.asarray(
            handle["sparse_direction_indices_zero_based"][:],
            dtype=np.int64,
        ).reshape(-1)
    np.testing.assert_array_equal(stored, q26_source_indices(Q26_CSV))


def test_hdf5_sparse_index_mismatch_is_rejected(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    dataset_root, split_csv = synthetic_dataset
    path = dataset_root / "subjects" / "P9001" / "q26.h5"
    with h5py.File(path, "r+") as handle:
        handle["sparse_direction_indices_zero_based"][0, 0] = 2
    with pytest.raises(ValueError, match="do not match"):
        build_q26_condition(dataset_root, split_csv, 9001, Q26_CSV)
