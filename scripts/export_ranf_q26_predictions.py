"""Export RANF test predictions from WSL using the frozen SONICOM IDs.

RANF requires contiguous internal listener IDs, while MCAR tables use the
original SONICOM IDs.  This script validates the 44-subject test mapping,
copies each predicted SOFA into the project artifact layout, and converts the
native RANF evaluation log to an original-ID per-subject CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


SUBJECT_EVALUATION = re.compile(r"(?:INFO:root:)?(P\d{4}) evaluation$")
EXPECTED_TEST_SUBJECTS = 44


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_test_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if len(rows) != EXPECTED_TEST_SUBJECTS:
        raise ValueError(f"Expected 44 test mapping rows, found {len(rows)}")
    internal = [row["internal_subject_id"].upper() for row in rows]
    original = [row["original_subject_id"].upper() for row in rows]
    if len(set(internal)) != len(rows) or len(set(original)) != len(rows):
        raise ValueError("RANF test mapping contains duplicate subject IDs")
    return rows


def parse_native_metrics(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    current: str | None = None
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = SUBJECT_EVALUATION.search(line)
            if match:
                current = match.group(1).upper()
                result[current] = {}
                continue
            if current is None:
                continue
            if "ITD difference" in line:
                result[current]["ITDDifference_us"] = float(line.rsplit(":", 1)[1])
            elif "ILD difference" in line:
                result[current]["ILDDifference_dB"] = float(line.rsplit(":", 1)[1])
            elif "LSD (dB)" in line:
                result[current]["LSD_dB"] = float(line.rsplit(":", 1)[1])
    required = {"ITDDifference_us", "ILDDifference_dB", "LSD_dB"}
    incomplete = {subject: sorted(required - values.keys()) for subject, values in result.items() if required - values.keys()}
    if len(result) != EXPECTED_TEST_SUBJECTS or incomplete:
        raise ValueError(
            f"Expected 44 complete native metric records, found {len(result)}; "
            f"incomplete={incomplete}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    mapping_rows = read_test_mapping(args.mapping)
    eval_log = args.eval_root / "eval.log"
    if not eval_log.is_file():
        raise FileNotFoundError(eval_log)
    native_metrics = parse_native_metrics(eval_log)

    output_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    for mapping in mapping_rows:
        internal = mapping["internal_subject_id"].upper()
        original = mapping["original_subject_id"].upper()
        source = args.eval_root / f"pred_{internal.lower()}.sofa"
        if not source.is_file():
            raise FileNotFoundError(source)
        subject_root = args.output_root / "subjects" / original
        subject_root.mkdir(parents=True, exist_ok=True)
        destination = subject_root / "prediction.sofa"
        shutil.copy2(source, destination)
        prediction_hash = sha256(destination)
        output_rows.append(
            {
                **mapping,
                "prediction_relative_path": str(destination.relative_to(args.output_root)),
                "prediction_sha256": prediction_hash,
            }
        )
        metrics = native_metrics[internal]
        native_rows.append(
            {
                "SubjectLabel": original,
                "SubjectID": int(original[1:]),
                "RANFInternalSubjectLabel": internal,
                **metrics,
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    mapping_output = args.output_root / "subject_mapping.csv"
    with mapping_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    native_output = args.output_root / "native_per_subject_metrics.csv"
    with native_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(native_rows[0]))
        writer.writeheader()
        writer.writerows(native_rows)

    exp_root = args.eval_root.parent.parent
    checkpoint_paths = [exp_root / "adaptation.ckpt", exp_root / "adaptation_loss.ckpt"]
    manifest = {
        "schema_version": "1.0",
        "method": "RANF",
        "split": "test",
        "subject_count": len(output_rows),
        "source_eval_root": str(args.eval_root),
        "source_mapping": str(args.mapping),
        "eval_log_sha256": sha256(eval_log),
        "checkpoint_sha256": {
            path.name: sha256(path) for path in checkpoint_paths if path.is_file()
        },
        "prediction_sha256": {
            str(row["original_subject_id"]): str(row["prediction_sha256"])
            for row in output_rows
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(output_rows)} RANF test predictions to {args.output_root}")


if __name__ == "__main__":
    main()
