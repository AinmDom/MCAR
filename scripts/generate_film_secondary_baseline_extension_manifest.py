"""Freeze the validation-only RANF/FSP-AE secondary-metric extension manifest."""

from __future__ import annotations

import csv
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def subject_labels(root: Path) -> list[str]:
    split_csv = root / "configs/data/sonicom_subject_split_v1.csv"
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = [row["subject_id"] for row in csv.DictReader(handle) if row["split"] == "val"]
    if len(labels) != 44 or len(set(labels)) != 44:
        raise ValueError("Expected 44 unique validation subjects")
    return labels


def inventory_digest(root: Path, subjects: list[str], filename: str) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        source = root / "subjects" / subject / filename
        digest.update(subject.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def flat_inventory_digest(root: Path, subjects: list[str], template: str) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        source = root / template.format(subject=subject)
        digest.update(subject.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def resource(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": file_sha256(root / relative)}


def main() -> None:
    root = project_root()
    labels = subject_labels(root)
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked:
        raise RuntimeError("Tracked working tree must be clean before manifest freeze")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()

    deferred_manifest = json.loads(
        (root / "configs/experiments/sonicom_film_deferred_secondary_metrics_validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    ranf_root = root / "artifacts/reconstruction/sonicom_ranf_q26_validation_frozen"
    fsp_root = root / "artifacts/reconstruction/sonicom_fsp_ae_q26_formal_validation"
    sofa_root_relative = "data/HRTF/sonicom_measured_ffcmp_minphase_44k1/subjects"
    sofa_template = "{subject}_FreeFieldCompMinPhase_44kHz.sofa"
    sofa_root = root / sofa_root_relative
    output_relative = "results/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation"
    if (root / output_relative).exists() or (root / f"{output_relative}.partial").exists():
        raise FileExistsError("Baseline-extension output already exists")

    payload = {
        "schema_version": "1.0",
        "status": "frozen",
        "created_on": "2026-08-30",
        "purpose": "Comparator-specific validation-only extension of frozen FiLM secondary metrics to RANF and FSP-AE",
        "protocol": resource(
            root,
            "experiments/film_siren/FUTURE_FILM_SECONDARY_BASELINE_EXTENSION_AMENDMENT_V1.md",
        ),
        "hierarchy": "secondary_or_exploratory_only; cannot alter frozen candidate",
        "dataset": {
            "root": "data/processed/sonicom_residual_q26_v1",
            "split_csv": "configs/data/sonicom_subject_split_v1.csv",
            "split": "val",
            "subject_count": 44,
            "direction_count": 793,
            "q26_direction_count": 26,
            "interpolation_direction_count": 767,
            "horizontal_interpolation_direction_count": 72,
            "frequency_bin_count": 463,
            "resources": [
                resource(root, "configs/data/sonicom_preparation_report_v1.json"),
                resource(root, "configs/data/sonicom_subject_split_v1.csv"),
                resource(root, "configs/data/sonicom_sparse_grid_q26_v1.csv"),
                resource(root, "data/processed/sonicom_residual_q26_v1/training_statistics.json"),
            ],
            "reference_sofa_root": sofa_root_relative,
            "reference_sofa_filename_template": sofa_template,
            "reference_sofa_inventory_sha256": flat_inventory_digest(
                sofa_root, labels, sofa_template
            ),
            "coordinate_tolerance_degrees": 1e-5,
        },
        "methods": {
            "RANF": {
                "label": "RANF",
                "prediction_root": "artifacts/reconstruction/sonicom_ranf_q26_validation_frozen",
                "filename": "prediction.sofa",
                "prediction_inventory_sha256": inventory_digest(
                    ranf_root, labels, "prediction.sofa"
                ),
                "provenance_path": "artifacts/reconstruction/sonicom_ranf_q26_validation_frozen/manifest.json",
                "provenance_sha256": file_sha256(ranf_root / "manifest.json"),
                "spectral_mapping": "FFT(Data.IR,n=1024); select source strict_ild zero-based bins",
            },
            "FSPAE": {
                "label": "FSP-AE",
                "prediction_root": "artifacts/reconstruction/sonicom_fsp_ae_q26_formal_validation",
                "filename": "prediction.h5",
                "prediction_inventory_sha256": inventory_digest(
                    fsp_root, labels, "prediction.h5"
                ),
                "provenance_path": "artifacts/reconstruction/sonicom_fsp_ae_q26_formal_validation/summary.json",
                "provenance_sha256": file_sha256(fsp_root / "summary.json"),
                "spectral_mapping": "predicted_magnitude_db source selected bin i maps to FSP bin i-1",
            },
        },
        "existing_results": {
            "bounded_secondary": resource(
                root, "results/sonicom_film_secondary_metrics_v1_validation/per_subject_metrics.csv"
            ),
            "bounded_spatial": resource(
                root, "results/sonicom_film_secondary_metrics_v1_validation/spatial_distance_bins.csv"
            ),
            "bounded_primary": resource(
                root, "results/sonicom_bounded_mcar_film_correction_final_e25_validation/metric_long.csv"
            ),
            "comparator_primary": resource(
                root, "results/sonicom_film_siren_gl_final_d1d2_notch_vs_ranf_fsp_v351_validation/metric_long.csv"
            ),
            "bounded_deferred": resource(
                root, "results/sonicom_film_deferred_secondary_metrics_v1_validation/per_subject_endpoints.csv"
            ),
        },
        "implementation": {
            "git_commit": head,
            "git_dirty_at_freeze": False,
            "python_executable": "D:/miniconda3/envs/ml/python.exe",
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "h5py": h5py.__version__,
                "scipy": scipy.__version__,
                "torch": str(torch.__version__),
            },
            "torchaudio_site_packages": deferred_manifest["implementation"][
                "torchaudio_site_packages"
            ],
            "torchaudio_version": deferred_manifest["implementation"]["torchaudio_version"],
            "resources": [
                resource(root, "scripts/evaluate_film_secondary_baseline_extension_validation.py"),
                resource(root, "src/mcar/evaluation/secondary_metrics.py"),
                resource(root, "src/mcar/evaluation/deferred_secondary_metrics.py"),
                resource(root, "src/mcar/fsp_ae_signal.py"),
                resource(root, "src/mcar/losses.py"),
                resource(root, "src/mcar/training/train_mlp_v2.py"),
                resource(root, "src/mcar/training/train_film_siren.py"),
                resource(root, "src/mcar/training/train_film_siren_stage_c.py"),
                resource(root, "tests/test_secondary_baseline_extension.py"),
            ],
            "external_resources": deferred_manifest["implementation"]["external_resources"],
        },
        "secondary_endpoints": [
            "FullSphereLSD",
            "HFFirstDifferenceMAE",
            "HFSecondDifferenceMAE",
            "MultiScaleNotchDepthMAE",
            "ERBBandILDMean",
            "ERB-band horizontal ILD profile",
            "four fixed distance-to-Q26 LSD bins",
        ],
        "deferred_endpoints": deferred_manifest["notch"] | {
            "itd": deferred_manifest["itd"],
            "reported": [
                "DominantNotchPenalizedMAE_Hz",
                "DominantNotchMatchedMAE_Hz",
                "DominantNotchMissRate",
                "DominantNotchSpuriousRate",
                "ReferenceNotchFraction",
                "ITDWeightedMAE_us",
                "ITDMaximumAbsoluteError_us",
            ],
        },
        "itd": deferred_manifest["itd"],
        "notch": deferred_manifest["notch"],
        "statistics": {
            "subject_count": 44,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260829,
            "tie_tolerance": 1e-12,
            "worst_decile_subject_count": 5,
            "paired_difference": "BOUNDED minus RANF/FSPAE; negative favors BOUNDED",
        },
        "availability": {
            "gate_and_correction": "not_applicable_no_common_internal_quantity",
            "efficiency": "not_available_no_same_hardware_benchmark",
            "localization": "not_run_dependency_unavailable",
        },
        "output_root": output_relative,
        "command": "D:\\miniconda3\\envs\\ml\\python.exe scripts/evaluate_film_secondary_baseline_extension_validation.py configs/experiments/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation_manifest.json",
        "test_policy": "Validation only; no test path may be constructed; reports record test_subject_count_read=0.",
        "result_blind_statement": "Endpoint formulas and existing four-method results were already known. No RANF/FSP-AE output for these added secondary/deferred endpoints was computed, aggregated, or inspected before this comparator-specific freeze. Previously committed primary RANF/FSP-AE metrics are used only for preregistered paired tail-risk rows.",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
    destination = root / "configs/experiments/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation_manifest.json"
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"path": str(destination), "identity": payload["identity_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
