"""Prepare and freeze the Stage A4 confirmation subject lock file.

Selects 32 train subjects that never participated in A2/A3 configuration
selection: candidates exclude the five A2 subjects, are stratified into
eight `Avg RMS dB (Free Field)` layers, and four subjects are drawn per
layer with a fixed seed (STAGE_A4_CONFIRMATION_PROTOCOL.md section 3).

Outputs (deterministic):

- ``configs/data/siren_a4_confirmation_subjects_v1.csv``;
- ``configs/data/siren_a4_preparation_report_v1.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root

SEED = 20260820
LAYER_COUNT = 8
PER_LAYER = 4
TARGET_COUNT = LAYER_COUNT * PER_LAYER
RMS_FIELD = "Avg RMS dB (Free Field)"
EXCLUDED_A2_SUBJECTS = {"P0289", "P0346", "P0085", "P0076", "P0010"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    root = project_root()
    metadata_path = (args.metadata or root / "data" / "HRTF" /
                     "sonicom_measured_ffcmp_minphase_44k1" /
                     "metadata_and_readme" / "metadata.csv").resolve()
    split_path = (args.split or root / "configs" / "data" /
                  "sonicom_subject_split_v1.csv").resolve()
    output_dir = (args.output_dir or root / "configs" / "data").resolve()
    for path in (metadata_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = {row["Subject"]: row for row in csv.DictReader(handle)}

    train_ids = [row["subject_id"] for row in split_rows if row["split"] == "train"]
    candidates: list[tuple[str, float]] = []
    for subject_id in train_ids:
        if subject_id in EXCLUDED_A2_SUBJECTS:
            continue
        record = metadata.get(subject_id)
        if record is None:
            continue
        raw = record.get(RMS_FIELD, "").strip()
        if not raw:
            continue
        try:
            candidates.append((subject_id, float(raw)))
        except ValueError:
            continue
    if len(candidates) < TARGET_COUNT:
        raise ValueError(
            f"Only {len(candidates)} usable candidates; need {TARGET_COUNT}"
        )
    candidates.sort(key=lambda item: (item[1], item[0]))
    layers = np.array_split([item[0] for item in candidates], LAYER_COUNT)
    rng = np.random.default_rng(args.seed)
    selected: list[dict[str, str | int | float]] = []
    for layer, members in enumerate(layers, start=1):
        picks = rng.choice(members, size=PER_LAYER, replace=False)
        for pick in picks:
            value = next(v for sid, v in candidates if sid == str(pick))
            selected.append(
                {
                    "layer": layer,
                    "subject_id": str(pick),
                    "rms_free_field_db": value,
                    "layer_members": int(len(members)),
                }
            )
    selected.sort(key=lambda row: (int(row["layer"]), row["subject_id"]))
    if len({row["subject_id"] for row in selected}) != TARGET_COUNT:
        raise ValueError("Duplicate subject selected")

    subjects_csv = output_dir / "siren_a4_confirmation_subjects_v1.csv"
    with subjects_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("layer", "subject_id", "rms_free_field_db",
                        "layer_members"),
        )
        writer.writeheader()
        writer.writerows(selected)

    report = {
        "protocol": "STAGE_A4_CONFIRMATION_PROTOCOL.md",
        "seed": args.seed,
        "target_count": TARGET_COUNT,
        "layer_count": LAYER_COUNT,
        "per_layer": PER_LAYER,
        "excluded_a2_subjects": sorted(EXCLUDED_A2_SUBJECTS),
        "candidate_pool_size": len(candidates),
        "selected_subjects": selected,
        "subjects_csv": str(subjects_csv),
        "subjects_csv_sha256": file_sha256(subjects_csv),
    }
    report_path = output_dir / "siren_a4_preparation_report_v1.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
