"""Freeze Bounded-E25 Q14/Q26/Q50 inputs and inference dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/experiments/sonicom_bounded_e25_input_direction_sensitivity_v1.json"
)
OUTPUT = ROOT / (
    "configs/experiments/"
    "sonicom_bounded_e25_input_direction_sensitivity_v1_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def identity(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def resource(path: str, purpose: str) -> dict[str, object]:
    absolute = ROOT / path
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return {
        "path": path,
        "sha256": sha256(absolute),
        "size_bytes": absolute.stat().st_size,
        "purpose": purpose,
    }


def main() -> None:
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Generate the formal manifest only from a clean worktree")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "preregistered" or config["dataset"]["split"] != "val":
        raise ValueError("Expected the validation-only preregistered config")
    split_rows = list(
        csv.DictReader(
            (ROOT / config["dataset"]["split_file"]).open(
                "r", encoding="utf-8-sig", newline=""
            )
        )
    )
    subjects = [row["subject_id"] for row in split_rows if row["split"] == "val"]
    if len(subjects) != 44:
        raise ValueError("Expected 44 validation subjects")
    artifact_root = ROOT / "artifacts/sparsity" / config["output_name"]
    status_path = artifact_root / "preparation_status.csv"
    status = list(csv.DictReader(status_path.open("r", encoding="utf-8-sig", newline="")))
    if len(status) != 132 or not all(
        row["Success"].strip().lower() in {"1", "true"} for row in status
    ):
        raise ValueError("Expected 132 successful preparation rows")

    inventory: list[dict[str, object]] = []
    q26_mca_max = 0.0
    q26_correction_max = 0.0
    for subject in subjects:
        for count in config["dataset"]["direction_counts"]:
            level_root = artifact_root / "subjects" / subject / f"q{count}"
            input_path = level_root / "model_input.h5"
            cache_path = level_root / "cache.mat"
            if not input_path.is_file() or not cache_path.is_file():
                raise FileNotFoundError(level_root)
            with h5py.File(input_path, "r") as handle:
                split = handle.attrs["split"]
                if isinstance(split, bytes):
                    split = split.decode()
                actual_count = int(np.asarray(handle.attrs["sparse_direction_count"]).item())
                mca = np.asarray(handle["mca_logmag_db"][:], dtype=np.float32)
                correction = np.asarray(handle["correction_logmag_db"][:], dtype=np.float32)
                if split != "val" or actual_count != count:
                    raise ValueError(f"Input provenance mismatch for {input_path}")
                if mca.shape != (2, 793, 463) or correction.shape != mca.shape:
                    raise ValueError(f"Input tensor shape mismatch for {input_path}")
                if not bool(np.all(np.isfinite(mca))) or not bool(
                    np.all(np.isfinite(correction))
                ):
                    raise FloatingPointError(input_path)
            if count == 26:
                source = ROOT / config["dataset"]["root"] / "subjects" / subject / "q26.h5"
                with h5py.File(source, "r") as handle:
                    source_mca = np.asarray(handle["mca_logmag_db"][:], dtype=np.float32)
                    source_correction = np.asarray(
                        handle["correction_logmag_db"][:], dtype=np.float32
                    )
                q26_mca_max = max(q26_mca_max, float(np.max(np.abs(mca - source_mca))))
                q26_correction_max = max(
                    q26_correction_max,
                    float(np.max(np.abs(correction - source_correction))),
                )
            inventory.append(
                {
                    "subject_label": subject,
                    "direction_count": count,
                    "model_input": relative(input_path),
                    "model_input_sha256": sha256(input_path),
                    "model_input_size_bytes": input_path.stat().st_size,
                    "cache": relative(cache_path),
                    "cache_sha256": sha256(cache_path),
                    "cache_size_bytes": cache_path.stat().st_size,
                    "shape": [2, 793, 463],
                    "finite": True,
                    "split": "val",
                }
            )
    tolerance = 1e-4
    if q26_mca_max > tolerance or q26_correction_max > tolerance:
        raise ValueError(
            "Q26 preparation reproduction failed: "
            f"MCA={q26_mca_max}, correction={q26_correction_max}"
        )

    model_manifest = ROOT / config["paper_main_model"]["manifest"]
    manifest = {
        "schema_version": "1.0",
        "status": "frozen_before_inference",
        "created_on": "2026-09-01",
        "experiment_config": relative(CONFIG),
        "experiment_config_sha256": sha256(CONFIG),
        "paper_main_model_manifest": relative(model_manifest),
        "paper_main_model_manifest_sha256": sha256(model_manifest),
        "paper_main_model_identity_sha256": config["paper_main_model"][
            "manifest_identity_sha256"
        ],
        "split": "val",
        "subject_count": 44,
        "direction_counts": [14, 26, 50],
        "prepared_case_count": len(inventory),
        "q26_input_reproduction": {
            "maximum_mca_abs_error_db": q26_mca_max,
            "maximum_correction_abs_error_db": q26_correction_max,
            "tolerance_db": tolerance,
            "passed": True,
        },
        "resources": [
            resource(config["protocol"], "result-blind protocol"),
            resource(config["dataset"]["sparse_grid_file"], "nested Q14/Q26/Q50 grid"),
            resource(config["dataset"]["sparse_grid_report"], "grid geometry report"),
            resource(
                "matlab/+mcar/prepare_sonicom_learned_sparsity.m",
                "current-Q MCA input preparation",
            ),
            resource(
                "src/mcar/evaluation/predict_bounded_e25_direction_sensitivity.py",
                "hash-guarded frozen-ensemble inference",
            ),
            resource(
                "matlab/+mcar/evaluate_bounded_e25_direction_sensitivity.m",
                "fixed-mask strict evaluator",
            ),
        ],
        "input_inventory": inventory,
        "git": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "dirty": False,
        },
        "test_subject_count_read": 0,
    }
    manifest["identity_sha256"] = identity(manifest)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(relative(OUTPUT))
    print(manifest["identity_sha256"])


if __name__ == "__main__":
    main()
