"""Freeze inputs and implementation for the ten-method supplementary test run."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch

from mcar.paths import project_root
from mcar.training.train_film_siren import split_subject_paths


OUTPUT = Path(
    "configs/experiments/sonicom_complete_ten_method_secondary_deferred_test_v1_manifest.json"
)
TORCHAUDIO_SITE = Path(r"D:\miniconda3\envs\asd\Lib\site-packages")
METHODS = {
    "SHOnly": {"label": "SH only"},
    "SUpDEqSH": {"label": "SUpDEq SH"},
    "SUpDEqNN": {"label": "SUpDEq NN"},
    "SUpDEqBary": {"label": "SUpDEq Barycentric"},
    "MCA": {"label": "MCA"},
    "MCARv351": {"label": "MCAR v3.5.1"},
    "FSPAE": {"label": "FSP-AE"},
    "RANF": {"label": "RANF"},
    "HYBRID": {"label": "Hybrid E190"},
    "BOUNDED": {"label": "Bounded E25"},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def inventory_digest(paths: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for key, path in sorted(paths, key=lambda item: item[0]):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest().upper()


def require_clean_git(root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Manifest generation requires clean Git:\n{}".format(status))
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    root = project_root()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(output)
    commit = require_clean_git(root)
    dataset_root = root / "data/processed/sonicom_residual_q26_v1"
    split_csv = root / "configs/data/sonicom_subject_split_v1.csv"
    subject_rows = split_subject_paths(dataset_root, split_csv, "test")
    labels = [row[1] for row in subject_rows]
    if len(labels) != 44 or len(set(labels)) != 44:
        raise ValueError("Expected 44 frozen test subjects")

    sofa_root = Path("data/HRTF/sonicom_measured_ffcmp_minphase_44k1/subjects")
    source_inventory = inventory_digest([(label, path) for _, label, path in subject_rows])
    sofa_inventory = inventory_digest([
        (label, root / sofa_root / (label + "_FreeFieldCompMinPhase_44kHz.sofa"))
        for label in labels
    ])
    prediction_specs = {
        "MCARv351": {
            "root": "artifacts/reconstruction/sonicom_q26_test_v351_previous30_b70",
            "filename": "prediction.h5",
        },
        "FSPAE": {
            "root": "artifacts/reconstruction/sonicom_fsp_ae_q26_locked_final_test",
            "filename": "prediction.h5",
        },
        "RANF": {
            "root": "artifacts/reconstruction/sonicom_ranf_q26_final_test",
            "filename": "prediction.sofa",
        },
        "HYBRID": {
            "root": "artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_test",
            "filename": "prediction.h5",
        },
        "BOUNDED": {
            "root": "artifacts/reconstruction/sonicom_bounded_mcar_film_correction_final_e25_ensemble_test",
            "filename": "prediction.h5",
        },
    }
    for spec in prediction_specs.values():
        prediction_root = root / spec["root"]
        spec["inventory_sha256"] = inventory_digest([
            (label, prediction_root / "subjects" / label / spec["filename"])
            for label in labels
        ])

    implementation_paths = [
        Path("experiments/film_siren/STAGE_E_COMPLETE_TEN_METHOD_SECONDARY_DEFERRED_TEST_PROTOCOL.md"),
        Path("matlab/+mcar/export_sonicom_classical_secondary_validation.m"),
        Path("scripts/evaluate_complete_ten_method_test_metrics.py"),
        Path("scripts/generate_complete_ten_method_test_manifest.py"),
        Path("src/mcar/evaluation/secondary_metrics.py"),
        Path("src/mcar/evaluation/deferred_secondary_metrics.py"),
        Path("src/mcar/fsp_ae_signal.py"),
        Path("src/mcar/losses.py"),
        Path("tests/test_complete_ten_method_test_metrics.py"),
    ]
    external_paths = [
        TORCHAUDIO_SITE / "torchaudio-2.8.0+cu128.dist-info" / "RECORD",
        TORCHAUDIO_SITE / "torchaudio" / "lib" / "_torchaudio.pyd",
    ]
    primary_path = Path(
        "results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/metric_long.csv"
    )
    payload = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_on": "2026-09-02",
        "purpose": "Post-lock supplementary test characterization for all ten frozen methods",
        "protocol": {
            "path": "experiments/film_siren/STAGE_E_COMPLETE_TEN_METHOD_SECONDARY_DEFERRED_TEST_PROTOCOL.md",
            "hierarchy": "secondary_or_exploratory_only",
            "untouched_test_confirmation": False,
        },
        "authorization": {
            "authorized": True,
            "user_text": "现在可以开始在test集上完成完整的十方法个指标评价了，已有的test结果可以复用，没有的补充",
            "recorded_at_local": "2026-09-02T13:57:53+08:00",
        },
        "dataset": {
            "root": "data/processed/sonicom_residual_q26_v1",
            "split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "split": "test",
            "subject_count": 44,
            "source_inventory_sha256": source_inventory,
            "reference_sofa_root": str(sofa_root).replace("\\", "/"),
            "reference_sofa_inventory_sha256": sofa_inventory,
            "directions": 793,
            "q26_directions": 26,
            "interpolation_directions": 767,
            "horizontal_interpolation_directions": 72,
            "frequency_bins": 463,
        },
        "methods": METHODS,
        "source_predictions": prediction_specs,
        "primary_metrics": {
            "metric_long_csv": str(primary_path).replace("\\", "/"),
            "sha256": file_sha256(root / primary_path),
            "subject_rows": 1760,
            "methods": 10,
            "endpoints": 4,
            "policy": "reuse exactly; do not recompute or replace",
        },
        "standardized_export": {
            "prediction_root": "artifacts/reconstruction/sonicom_complete_ten_method_secondary_test_inputs_v1",
            "evaluation_root": "artifacts/evaluation/sonicom_complete_ten_method_secondary_test_export_v1",
            "metric_long_csv": "artifacts/evaluation/sonicom_complete_ten_method_secondary_test_export_v1/metric_long.csv",
            "expected_methods": 8,
            "expected_prediction_files": 352,
            "primary_reproduction_tolerance_db": 1e-9,
            "command": "mcar.export_sonicom_classical_secondary_validation(inf,'sonicom_complete_ten_method_secondary_test_export_v1',false,'test','sonicom_q26_test_v351_previous30_b70',true,'sonicom_fsp_ae_q26_locked_final_test','sonicom_ranf_q26_final_test','sonicom_complete_ten_method_secondary_test_inputs_v1')",
        },
        "implementation": {
            "git_commit": commit,
            "git_dirty_at_freeze": False,
            "python_executable": "D:/miniconda3/envs/ml/python.exe",
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
        "itd": {
            "sampling_rate_hz": 44100.0,
            "upsampled_rate_hz": 384000.0,
            "lowpass_hz": 1600.0,
            "maximum_itd_seconds": 0.001,
            "direction_batch_size": 64,
        },
        "notch": {
            "range_hz": [4000.0, 18000.0],
            "smoothing_window_bins": 11,
            "smoothing_degree": 3,
            "minimum_prominence_db": 1.0,
            "minimum_separation_hz": 500.0,
            "matching_tolerance_hz": 1500.0,
        },
        "bootstrap": {
            "replicates": 10000,
            "generator": "NumPy PCG64",
            "base_seed": 20260902,
            "subject_is_independent_unit": True,
        },
        "outputs": {
            "supplementary_results_root": "results/sonicom_complete_ten_method_secondary_deferred_test_v1",
            "complete_results_root": "results/sonicom_complete_ten_method_test_v1",
            "delivery_root": "outputs/sonicom_complete_ten_method_test_v1",
        },
        "commands": {
            "evaluate": "D:/miniconda3/envs/ml/python.exe scripts/evaluate_complete_ten_method_test_metrics.py configs/experiments/sonicom_complete_ten_method_secondary_deferred_test_v1_manifest.json --allow-test",
        },
        "expected_counts": {
            "subjects": 44, "methods": 10, "new_endpoints": 16,
            "new_per_subject_rows": 7040, "new_aggregate_rows": 160,
            "new_paired_rows": 144, "band_profile_rows": 350,
            "spatial_direction_rows": 7670, "complete_endpoints": 20,
            "complete_per_subject_rows": 8800, "complete_aggregate_rows": 200,
        },
        "test_policy": "Read exactly the frozen 44-subject test cohort; no inference, tuning, selection, or registry reset.",
        "result_blind_statement": "No new secondary/deferred test endpoint was parsed or aggregated before this freeze.",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "identity_sha256": payload["identity_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
