"""Prepare and freeze the Stage A2 five-subject matrix lock files.

Generates two committed CSV lock files plus a preparation report:

- ``configs/data/siren_a2_subjects_v1.csv``: five fixed train subjects
  selected by `Avg RMS dB (Free Field)` quintile stratification with a
  fixed seed, per STAGE_A2_PROTOCOL.md section 4.
- ``configs/data/siren_a2_holdout_v1.csv``: 64 fixed interpolation-only
  directions selected by solid-angle stratification with a fixed seed,
  shared by every subject and configuration (protocol section 5).

The script is deterministic: rerunning it must reproduce the same files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root

A2_SEED = 20260819
HOLDOUT_COUNT = 64
STRATUM_COUNT = 5
RMS_FIELD = "Avg RMS dB (Free Field)"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_split_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_metadata(metadata_csv: Path) -> dict[str, dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["Subject"]: row for row in csv.DictReader(handle)}


def select_subjects(
    split_rows: list[dict[str, str]],
    metadata: dict[str, dict[str, str]],
    seed: int,
) -> list[dict[str, str | int | float]]:
    train_ids = [row["subject_id"] for row in split_rows if row["split"] == "train"]
    candidates: list[tuple[str, float]] = []
    for subject_id in train_ids:
        record = metadata.get(subject_id)
        if record is None:
            continue
        raw = record.get(RMS_FIELD, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        candidates.append((subject_id, value))
    if len(candidates) < STRATUM_COUNT:
        raise ValueError(
            f"Only {len(candidates)} subjects have a usable {RMS_FIELD} value"
        )
    candidates.sort(key=lambda item: (item[1], item[0]))
    counts = np.array_split([item[0] for item in candidates], STRATUM_COUNT)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, str | int | float]] = []
    for stratum, members in enumerate(counts, start=1):
        pick = str(rng.choice(members))
        value = next(v for sid, v in candidates if sid == pick)
        selected.append(
            {
                "stratum": stratum,
                "subject_id": pick,
                "rms_free_field_db": value,
                "stratum_members": int(len(members)),
            }
        )
    selected.sort(key=lambda row: int(row["stratum"]))
    return selected


def select_holdout(
    reference_grid_path: Path,
    count: int,
    seed: int,
) -> tuple[list[int], str]:
    with reference_grid_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    interpolation = [
        row
        for row in rows
        if row["is_interpolation_evaluation"] == "1"
    ]
    if len(interpolation) < count:
        raise ValueError(
            f"Only {len(interpolation)} interpolation directions; need {count}"
        )
    weights = np.asarray(
        [float(row["solid_angle_weight"]) for row in interpolation],
        dtype=np.float64,
    )
    if not np.all(weights > 0.0) or not np.isfinite(weights).all():
        raise ValueError("Reference grid contains invalid solid-angle weights")
    normalized = weights / weights.sum()
    rng = np.random.default_rng(seed)
    # Stratified sampling without replacement by normalized solid-angle weight.
    chosen = np.zeros(count, dtype=np.int64)
    remaining = np.arange(len(interpolation), dtype=np.int64)
    remaining_weights = normalized.copy()
    for position in range(count):
        pick = int(rng.choice(remaining, p=remaining_weights / remaining_weights.sum()))
        chosen[position] = pick
        mask = remaining != pick
        remaining = remaining[mask]
        remaining_weights = remaining_weights[mask]
    holdout_source_indices = sorted(
        int(interpolation[int(index)]["source_index_zero_based"])
        for index in chosen
    )
    payload = json.dumps(holdout_source_indices).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest().upper()
    return holdout_source_indices, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--reference-grid", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=A2_SEED)
    args = parser.parse_args()

    root = project_root()
    metadata_path = (args.metadata or root / "data" / "HRTF" /
                     "sonicom_measured_ffcmp_minphase_44k1" /
                     "metadata_and_readme" / "metadata.csv").resolve()
    split_path = (args.split or root / "configs" / "data" /
                  "sonicom_subject_split_v1.csv").resolve()
    grid_path = (args.reference_grid or root / "configs" / "data" /
                 "sonicom_reference_grid_v1.csv").resolve()
    output_dir = (args.output_dir or root / "configs" / "data").resolve()
    for path in (metadata_path, split_path, grid_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    split_rows = read_split_rows(split_path)
    metadata = read_metadata(metadata_path)
    subjects = select_subjects(split_rows, metadata, args.seed)
    holdout_indices, holdout_sha256 = select_holdout(
        grid_path, HOLDOUT_COUNT, args.seed
    )

    subjects_csv = output_dir / "siren_a2_subjects_v1.csv"
    with subjects_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("stratum", "subject_id",
                                "rms_free_field_db", "stratum_members")
        )
        writer.writeheader()
        writer.writerows(subjects)

    holdout_csv = output_dir / "siren_a2_holdout_v1.csv"
    with holdout_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("version", "holdout_count", "seed",
                        "source_indices_zero_based", "sha256"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "version": "siren_a2_holdout_v1",
                "holdout_count": HOLDOUT_COUNT,
                "seed": args.seed,
                "source_indices_zero_based": ",".join(
                    str(index) for index in holdout_indices
                ),
                "sha256": holdout_sha256,
            }
        )

    report = {
        "protocol": "STAGE_A2_PROTOCOL.md",
        "seed": args.seed,
        "stratum_field": RMS_FIELD,
        "selected_subjects": subjects,
        "holdout_count": HOLDOUT_COUNT,
        "holdout_sha256": holdout_sha256,
        "subjects_csv": str(subjects_csv),
        "subjects_csv_sha256": file_sha256(subjects_csv),
        "holdout_csv": str(holdout_csv),
        "holdout_csv_sha256": file_sha256(holdout_csv),
    }
    report_path = output_dir / "siren_a2_preparation_report_v1.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
