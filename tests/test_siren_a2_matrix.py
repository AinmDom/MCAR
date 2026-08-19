"""Unit tests for the Stage A2 matrix lock-file readers.

The tests use a workspace-local scratch directory instead of the standard
library TemporaryDirectory, which is denied by the DSH file sandbox.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from mcar.training.train_siren import load_holdout, load_subject_ids

SCRATCH = Path("artifacts") / "_siren_a2_test_tmp"


@pytest.fixture()
def scratch_dir() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        yield SCRATCH
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def make_subjects_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("stratum", "subject_id", "rms_free_field_db",
                                "stratum_members")
        )
        writer.writeheader()
        for stratum, subject_id in enumerate(
            ("P0289", "P0346", "P0085", "P0076", "P0010"), start=1
        ):
            writer.writerow(
                {
                    "stratum": stratum,
                    "subject_id": subject_id,
                    "rms_free_field_db": -40.0 + stratum,
                    "stratum_members": 52,
                }
            )


def test_load_subject_ids_returns_numeric_ids(scratch_dir: Path) -> None:
    path = scratch_dir / "subjects.csv"
    make_subjects_csv(path)
    assert load_subject_ids(path) == [289, 346, 85, 76, 10]


def test_load_subject_ids_rejects_bad_label(scratch_dir: Path) -> None:
    path = scratch_dir / "subjects.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject_id",))
        writer.writeheader()
        writer.writerow({"subject_id": "X123"})
    with pytest.raises(ValueError, match="subject label"):
        load_subject_ids(path)


def test_load_subject_ids_rejects_empty(scratch_dir: Path) -> None:
    path = scratch_dir / "subjects.csv"
    path.write_text("subject_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_subject_ids(path)


def make_holdout_csv(path: Path, indices: list[int]) -> tuple[list[int], str]:
    payload = json.dumps(indices).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest().upper()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("version", "holdout_count", "seed",
                        "source_indices_zero_based", "sha256"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "version": "siren_a2_holdout_v1",
                "holdout_count": len(indices),
                "seed": 20260819,
                "source_indices_zero_based": ",".join(str(i) for i in indices),
                "sha256": digest,
            }
        )
    return indices, digest


def test_load_holdout_round_trip(scratch_dir: Path) -> None:
    path = scratch_dir / "holdout.csv"
    indices, digest = make_holdout_csv(path, [2, 3, 12, 79])
    loaded, loaded_digest = load_holdout(path)
    np.testing.assert_array_equal(
        loaded, np.asarray(indices, dtype=np.int64)
    )
    assert loaded_digest == digest


def test_load_holdout_detects_tampering(scratch_dir: Path) -> None:
    path = scratch_dir / "holdout.csv"
    make_holdout_csv(path, [2, 3, 12, 79])
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("2,3,12,79", "2,3,12,80"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_holdout(path)


def test_load_holdout_rejects_duplicates(scratch_dir: Path) -> None:
    path = scratch_dir / "holdout.csv"
    make_holdout_csv(path, [5, 5])
    with pytest.raises(ValueError, match="unique"):
        load_holdout(path)
