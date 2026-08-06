"""Prepare ignored SONICOM caches for the FSP-AE Q26 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcar.fsp_ae_data import (
    calculate_normalization,
    cache_path,
    load_sonicom_sofa,
    resolve_sofa_path,
    subject_ids_for_split,
    write_normalization,
    write_subject_cache,
)
from mcar.paths import project_root


EXPECTED_COUNTS = {"train": 262, "val": 44, "test": 44}


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sofa-root",
        type=Path,
        default=root
        / "data"
        / "HRTF"
        / "sonicom_measured_ffcmp_minphase_44k1"
        / "subjects",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=root / "configs" / "data" / "sonicom_subject_split_v1.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "data" / "processed" / "sonicom_fsp_ae_q26_v1",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val")
    )
    parser.add_argument("--subject-limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.subject_limit is not None and args.subject_limit <= 0:
        raise ValueError("subject-limit must be positive")
    if "test" in args.splits and not args.allow_test:
        raise PermissionError("--splits test requires --allow-test")

    prepared: dict[str, list[Path]] = {}
    for split in args.splits:
        subject_ids = subject_ids_for_split(
            args.split_csv, split, allow_test=args.allow_test
        )
        if args.subject_limit is not None:
            subject_ids = subject_ids[: args.subject_limit]
        prepared[split] = []
        for subject_index, subject_id in enumerate(subject_ids, start=1):
            output_path = cache_path(args.output_root, split, subject_id)
            if output_path.is_file() and not args.overwrite:
                prepared[split].append(output_path)
                print(
                    f"[{split} {subject_index}/{len(subject_ids)}] reuse {output_path}"
                )
                continue
            sofa_path = resolve_sofa_path(args.sofa_root, subject_id)
            print(
                f"[{split} {subject_index}/{len(subject_ids)}] prepare {subject_id}"
            )
            data = load_sonicom_sofa(sofa_path, subject_id, split)
            write_subject_cache(data, output_path)
            prepared[split].append(output_path)

    train_files = prepared.get("train")
    if train_files:
        statistics = calculate_normalization(train_files)
        write_normalization(
            args.output_root / "normalization.json",
            statistics,
            EXPECTED_COUNTS["train"],
        )
        print(
            {
                "status": "completed",
                "prepared": {key: len(value) for key, value in prepared.items()},
                "normalization": statistics,
                "test_subject_count_read": (
                    len(prepared.get("test", [])) if args.allow_test else 0
                ),
            }
        )


if __name__ == "__main__":
    main()
