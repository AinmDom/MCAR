"""Offline checks for frozen Q26 parsing and test-access refusal."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from mcar.fsp_ae_data import subject_ids_for_split


def main() -> None:
    with TemporaryDirectory() as directory:
        split_file = Path(directory) / "split.csv"
        with split_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("subject_id", "split"))
            writer.writeheader()
            writer.writerows(
                (
                    {"subject_id": "P0001", "split": "val"},
                    {"subject_id": "P0002", "split": "train"},
                    {"subject_id": "P0003", "split": "test"},
                )
            )
        assert subject_ids_for_split(split_file, "train") == ["P0002"]
        assert subject_ids_for_split(split_file, "validation") == ["P0001"]
        try:
            subject_ids_for_split(split_file, "test")
        except PermissionError:
            pass
        else:
            raise AssertionError("test split was accessible without allow_test")
        assert subject_ids_for_split(
            split_file, "test", allow_test=True
        ) == ["P0003"]
    print({"status": "passed", "test_access_default": "refused"})


if __name__ == "__main__":
    main()
