"""Freeze secondary validation comparison of LSD-B versus existing Hybrid."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/experiments/sonicom_hybrid_lsd_b_secondary_validation_manifest.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def identity(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def inventory(root: Path, subjects: list[str]) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        path = root / "subjects" / subject / "prediction.h5"
        digest.update(subject.encode("ascii") + b"\0" + sha(path).encode("ascii") + b"\n")
    return digest.hexdigest().upper()


def resource(path: str) -> dict:
    return {"path": path, "sha256": sha(ROOT / path)}


def method(label: str, prediction_root: str, identity_value: str, subjects: list[str]) -> dict:
    report = f"{prediction_root}/inference_report.json"
    return {"label": label, "prediction_root": prediction_root, "inference_report": report,
        "inference_report_sha256": sha(ROOT / report),
        "prediction_inventory_sha256": inventory(ROOT / prediction_root, subjects), "identity": identity_value}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    split_rows = (ROOT / "configs/data/sonicom_subject_split_v1.csv").read_text().splitlines()[1:]
    subjects = [row.split(",")[0] for row in split_rows if row.split(",")[1] == "val"]
    if len(subjects) != 44:
        raise ValueError("Expected 44 validation subjects")
    protocol = "experiments/film_siren/STAGE_D_HYBRID_LSD_B_E190_PROTOCOL.md"
    evaluator = "scripts/evaluate_film_secondary_metrics_validation.py"
    primary = "results/sonicom_hybrid_lsd_b_vs_old_hybrid_validation/metric_long.csv"
    new_manifest = json.loads((ROOT / "configs/experiments/sonicom_hybrid_lsd_b_e190_ensemble_manifest.json").read_text())
    old_manifest = json.loads((ROOT / "configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json").read_text())
    film_manifest = json.loads((ROOT / "configs/experiments/sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_manifest.json").read_text())
    payload = {"schema_version": "1.0", "status": "frozen", "created_on": "2026-09-05",
        "purpose": "Pre-registered validation comparison of Hybrid LSD-B against existing Hybrid and frozen context methods",
        "protocol": {**resource(protocol), "hierarchy": "exploratory_validation_development"},
        "dataset": {"root": "data/processed/sonicom_residual_q26_v1", "split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "split": "val", "subject_count": 44, "direction_count": 793, "q26_direction_count": 26,
            "interpolation_direction_count": 767, "horizontal_interpolation_direction_count": 72,
            "frequency_bin_count": 463, "resources": [resource(path) for path in (
                "configs/data/sonicom_preparation_report_v1.json", "configs/data/sonicom_subject_split_v1.csv",
                "configs/data/sonicom_sparse_grid_q26_v1.csv", "data/processed/sonicom_residual_q26_v1/training_statistics.json")]},
        # The evaluator's legacy key BOUNDED is the candidate slot. Labels and
        # identities below are authoritative and prevent semantic ambiguity.
        "methods": {
            "BOUNDED": method("Hybrid LSD B E190 ensemble", "artifacts/reconstruction/sonicom_hybrid_lsd_b_e190_ensemble_validation", new_manifest["identity_sha256"], subjects),
            "HYBRID": method("Original Hybrid E190 ensemble", "artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_validation", old_manifest["identity_sha256"], subjects),
            "FILMENS": method("FiLM-SIREN corrected E130 ensemble", "artifacts/reconstruction/sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_validation", film_manifest["identity_sha256"], subjects),
            "MCAR": method("MCAR v3.5.1", "artifacts/reconstruction/sonicom_q26_validation_v351_previous30_b70", "v3.5.1-candidate", subjects)},
        "method_alias": {"BOUNDED": "HYBRID_LSD_B candidate", "HYBRID": "existing HYBRID baseline"},
        "primary_metrics": {"metric_long_csv": primary, "sha256": sha(ROOT / primary),
            "use": "paired tail statistics; generated on the same validation predictions",
            "method_mapping": {"HYBRID": "BOUNDED", "PARENT": "HYBRID", "FILMENS": "FILMENS", "MCAR": "MCAR"}},
        "implementation": {"git_commit": "4d487b0bd032a8ec8391c66738b4479757d1973c", "git_dirty_at_freeze": True,
            "implementation_hash_locked": True,
            "python_executable": "D:/miniconda3/envs/ml/python.exe",
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__, "torch": str(torch.__version__)},
            "bootstrap_generator": "numpy.random.default_rng PCG64",
            "resources": [resource(path) for path in (evaluator, "src/mcar/evaluation/secondary_metrics.py",
                "src/mcar/losses.py", "src/mcar/training/train_mlp_v2.py", "src/mcar/training/train_film_siren.py",
                "src/mcar/training/train_film_siren_stage_c.py", "tests/test_secondary_metrics.py")]},
        "registered_tranche": {"scalar_endpoints": ["FullSphereLSD", "HFFirstDifferenceMAE", "HFSecondDifferenceMAE", "MultiScaleNotchDepthMAE", "ERBBandILDMean"],
            "profiles": ["ERB-band horizontal ILD absolute-error profile", "LSD by fixed distance-to-Q26 bin", "subject and direction LSD maps"],
            "paired_tail_risk": "all five scalar endpoints, four primary endpoints, and four distance-bin endpoints",
            "bootstrap_replicates": 10000, "bootstrap_seed": 20260904, "tie_tolerance": 1e-12, "worst_decile_subject_count": 5},
        "deferred_unqualified_endpoints": ["model_based_localization", "ITD_sanity", "gate_and_correction_behavior", "efficiency_and_deployability", "discrete_notch_location"],
        "output_root": "results/sonicom_hybrid_lsd_b_secondary_validation",
        "test_policy": "Validation only. The entry point rejects non-val sources and records test_subject_count_read=0."}
    payload["identity_sha256"] = identity(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT.relative_to(ROOT).as_posix(), "identity_sha256": payload["identity_sha256"]}))


if __name__ == "__main__":
    main()
