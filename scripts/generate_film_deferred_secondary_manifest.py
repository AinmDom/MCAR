"""Freeze the result-blind deferred secondary-metrics validation manifest."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch

from mcar.paths import project_root
from mcar.training.train_film_siren import file_sha256, split_subject_paths


OUTPUT = Path("configs/experiments/sonicom_film_deferred_secondary_metrics_validation_manifest.json")
BASE_MANIFEST = Path("configs/experiments/sonicom_film_secondary_metrics_v1_validation_manifest.json")
BOUNDED_MANIFEST = Path(
    "configs/experiments/sonicom_bounded_mcar_film_correction_final_e25_ensemble_manifest.json"
)
TORCHAUDIO_SITE = Path(r"D:\miniconda3\envs\asd\Lib\site-packages")


def require_clean_git(root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"Manifest generation requires clean Git:\n{status}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def inventory_digest(paths: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for label, path in sorted(paths):
        digest.update(label.encode("ascii") + b"\0")
        digest.update(file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest().upper()


def main() -> None:
    root = project_root()
    if (root / OUTPUT).exists():
        raise FileExistsError(root / OUTPUT)
    commit = require_clean_git(root)
    base = json.loads((root / BASE_MANIFEST).read_text(encoding="utf-8"))
    bounded = json.loads((root / BOUNDED_MANIFEST).read_text(encoding="utf-8"))
    subject_rows = split_subject_paths(
        root / base["dataset"]["root"], root / base["dataset"]["split_csv"], "val"
    )
    labels = [row[1] for row in subject_rows]
    sofa_root = Path("data/HRTF/sonicom_measured_ffcmp_minphase_44k1/subjects")
    sofa_paths = [
        (label, root / sofa_root / f"{label}_FreeFieldCompMinPhase_44kHz.sofa")
        for label in labels
    ]
    implementation_paths = [
        Path("scripts/evaluate_film_deferred_secondary_metrics_validation.py"),
        Path("scripts/benchmark_bounded_mcar_film_efficiency.py"),
        Path("src/mcar/evaluation/deferred_secondary_metrics.py"),
        Path("src/mcar/evaluation/secondary_metrics.py"),
        Path("src/mcar/fsp_ae_signal.py"),
        Path("src/mcar/predictors.py"),
        Path("tests/test_deferred_secondary_metrics.py"),
    ]
    external_paths = [
        TORCHAUDIO_SITE / "torchaudio-2.8.0+cu128.dist-info" / "RECORD",
        TORCHAUDIO_SITE / "torchaudio" / "lib" / "_torchaudio.pyd",
    ]
    gpu_row = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    gpu_name, driver_version, power_limit = (item.strip() for item in gpu_row)
    payload = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_on": "2026-08-30",
        "purpose": "Validation-only gate, ITD, dominant-notch, and efficiency completion tranche",
        "protocol": {
            "parent": base["protocol"],
            "amendment": "experiments/film_siren/FUTURE_FILM_DEFERRED_SECONDARY_METRICS_AMENDMENT_V1.md",
            "hierarchy": "secondary_or_descriptive_only",
        },
        "dataset": {
            **base["dataset"],
            "reference_sofa_root": str(sofa_root).replace("\\", "/"),
            "reference_sofa_inventory_sha256": inventory_digest(sofa_paths),
            "coordinate_tolerance_degrees": 1e-5,
        },
        "methods": base["methods"],
        "bounded_manifest": {
            "path": str(BOUNDED_MANIFEST).replace("\\", "/"),
            "identity_sha256": bounded["identity_sha256"],
        },
        "implementation": {
            "git_commit": commit,
            "git_dirty_at_freeze": False,
            "python_executable": r"D:/miniconda3/envs/ml/python.exe",
            "runtime_versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "h5py": h5py.__version__, "scipy": scipy.__version__,
                "torch": str(torch.__version__),
            },
            "torchaudio_site_packages": str(TORCHAUDIO_SITE),
            "torchaudio_version": "2.8.0+cu128",
            "resources": [
                {"path": str(path).replace("\\", "/"), "sha256": file_sha256(root / path)}
                for path in implementation_paths
            ],
            "external_resources": [
                {"path": str(path), "sha256": file_sha256(path)} for path in external_paths
            ],
        },
        "gate": {
            "directions_per_block": 32,
            "prediction_tolerance_db": 5e-5,
            "saved_full_grid": "all 793 directions; aggregate summaries use only 767 interpolation directions",
            "distance_bins_degrees": [[0, 10], [10, 20], [20, 30], [30, 180]],
            "ensemble_rule": "weighted mean of applied member corrections; no ensemble gate",
            "spearman_hierarchy": "one rho per subject over ear x interpolation direction x frequency",
        },
        "itd": {
            "estimator": "mcar.fsp_ae_signal.estimate_itd_seconds",
            "origin": "public compatibility port of official FSP-AE low-pass/cross-correlation procedure",
            "sampling_rate_hz": 44100.0,
            "upsampled_rate_hz": 384000.0,
            "lowpass_hz": 1600.0,
            "maximum_itd_seconds": 0.001,
            "direction_batch_size": 64,
            "interpolation": "none after argmax; resolution is 1/384000 s",
            "reference": "measured subject SOFA Data.IR evaluated by the same estimator",
            "candidate": "strict HRIR with inherited MCA phase and outside-band complex bins",
        },
        "notch": {
            "representation": "ear-specific DTF after subtracting solid-angle-weighted directional mean",
            "smoothing": "Savitzky-Golay, 11 bins, degree 3, scipy mode=interp",
            "range_hz": [4000.0, 18000.0],
            "minimum_prominence_db": 1.0,
            "minimum_separation_hz": 500.0,
            "selection": "single highest-prominence notch; ties choose lowest frequency",
            "matching_tolerance_hz": 1500.0,
            "unmatched_penalty_hz": 1500.0,
        },
        "efficiency": {
            "subject_label": labels[0],
            "batch_size": 1,
            "directions_per_block": 32,
            "warmups": 5,
            "repeats": 30,
            "precision": "float32",
            "cpu_threads": 1,
            "tf32": False,
            "hardware": {
                "gpu_name": gpu_name,
                "driver_version": driver_version,
                "power_limit_w": float(power_limit),
                "power_mode": "driver-managed default; no application clock or power override",
                "cpu": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "operating_system": platform.platform(),
            },
            "software": {
                "python": platform.python_version(), "torch": str(torch.__version__),
                "cuda_runtime": str(torch.version.cuda),
                "cudnn": str(torch.backends.cudnn.version()),
            },
            "flop_profiler": "torch.utils.flop_counter.FlopCounterMode 2.8.0",
            "flop_convention": "native profiler FLOPs; MACs=FLOPs/2",
            "latency_scope": "prepared CPU arrays through GPU transfer/model/CPU result; excludes checkpoint and disk I/O",
            "end_to_end_scope": "input HDF5 load, three-member prediction, averaging, output HDF5 serialization",
        },
        "localization": {
            "status": "not_runnable_dependency_unavailable",
            "reason": "No immutable AMT/SAM package or published model is present locally; no output may be generated",
            "local_inventory": ["external/SUpDEq"],
        },
        "statistics": {
            "subject_count": 44, "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260829, "tie_tolerance": 1e-12,
            "candidate_vs_itd_baseline": "BOUNDED minus MCAR",
        },
        "outputs": {
            "results_root": "results/sonicom_film_deferred_secondary_metrics_v1_validation",
            "diagnostics_root": "artifacts/reconstruction/sonicom_bounded_mcar_film_gate_diagnostics_v1_validation",
            "efficiency_root": "results/sonicom_bounded_mcar_film_efficiency_v1_validation",
        },
        "commands": [
            r"D:\miniconda3\envs\ml\python.exe scripts/evaluate_film_deferred_secondary_metrics_validation.py configs/experiments/sonicom_film_deferred_secondary_metrics_validation_manifest.json",
            r"D:\miniconda3\envs\ml\python.exe scripts/benchmark_bounded_mcar_film_efficiency.py configs/experiments/sonicom_film_deferred_secondary_metrics_validation_manifest.json",
        ],
        "expected_counts": {
            "subjects": 44, "methods": 4, "directions": 793,
            "interpolation_directions": 767, "frequency_bins": 463,
            "bounded_members": 3,
        },
        "test_policy": "Validation only; non-val inputs are rejected and test_subject_count_read remains zero.",
        "result_blind_statement": "No deferred validation endpoint or efficiency result was inspected before this freeze.",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
    (root / OUTPUT).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "identity_sha256": payload["identity_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
