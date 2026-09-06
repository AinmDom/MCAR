"""Evaluate LAP Challenge 2024 Task 2 metrics on the frozen SONICOM test set.

The metric kernels below intentionally mirror Spatial Audio Metrics 0.0.8:
FFT-based HRTF magnitudes for LSD, broadband HRIR RMS ILD, and the MAXIACCe
ITD estimator with the official ``idx_lag - hrir_length`` offset.  The 0.0.8
wheel contains a Python list-division typo in its final ITD conversion; only a
``np.asarray`` type conversion is applied here, without changing the algorithm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import h5py
import numpy as np
import scipy
from scipy.fft import fft, fftfreq
from scipy.signal import butter, correlate, hilbert, lfilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcar.evaluation.deferred_secondary_metrics import strict_hrir_from_selected_db
from scripts.evaluate_complete_ten_method_test_metrics import (
    METHODS,
    STANDARDIZED_METHOD_PATHS,
    decode_attribute,
    file_sha256,
    inventory_digest,
    load_residual_prediction,
    load_standardized_prediction,
    strict_metadata,
    test_subject_paths,
)


METRIC_NAMES = ("LAP2024LSD_dB", "LAP2024ILDMAE_dB", "LAP2024ITDMAE_us")
THRESHOLDS = {
    "LAP2024LSD_dB": 7.4,
    "LAP2024ILDMAE_dB": 4.4,
    "LAP2024ITDMAE_us": 100.0,
}
STANDARDIZED_METHOD_PATHS = dict(STANDARDIZED_METHOD_PATHS)
EXPECTED_WHEEL_SHA256 = "1BEF9BBCE1CCEFA0520C416DB3D6E3A665F50E64E9996CA2C916688400149B7B"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--sam-wheel", type=Path, required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--compatibility-only", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Inconsistent columns: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def manifest_identity(payload: Mapping[str, Any]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256")).upper()
    actual = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Frozen source manifest identity mismatch")
    return actual


def wheel_members_sha256(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest().upper()
            for name in archive.namelist()
            if name.startswith("spatialaudiometrics/") and name.endswith(".py")
        }


def load_official_sam(wheel: Path):
    wheel_hash = file_sha256(wheel)
    if wheel_hash != EXPECTED_WHEEL_SHA256:
        raise ValueError(f"SAM wheel hash mismatch: {wheel_hash}")
    temp_dir = tempfile.TemporaryDirectory(prefix="lap_sam_0_0_8_")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(temp_dir.name)
    sys.path.insert(0, temp_dir.name)
    for module_name in list(sys.modules):
        if module_name == "spatialaudiometrics" or module_name.startswith("spatialaudiometrics."):
            del sys.modules[module_name]
    from spatialaudiometrics import hrtf_metrics as sam_hrtf_metrics

    return sam_hrtf_metrics, temp_dir


def sam_hrir2hrtf(hrir: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact SAM 0.0.8 hrir2hrtf magnitude/frequency behavior."""
    values = np.asarray(hrir)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("HRIR must have shape [location, ear, sample]")
    size = [values.shape[0], values.shape[1], int(values.shape[2] / 2)]
    hrtfs = np.empty(size)
    for location, loc in enumerate(values):
        for ear, signal in enumerate(loc):
            spectrum = fft(signal)
            frequencies = fftfreq(len(signal), 1.0 / fs)[: len(signal) // 2]
            hrtfs[location, ear, :] = np.abs(spectrum[0 : len(signal) // 2])
    return hrtfs, frequencies


def lap2024_lsd(hrir_reference: np.ndarray, hrir_estimate: np.ndarray, fs: float) -> float:
    if hrir_reference.shape != hrir_estimate.shape:
        raise ValueError("Reference and estimate HRIR shapes differ")
    reference, frequencies = sam_hrir2hrtf(hrir_reference, fs)
    estimate, estimate_frequencies = sam_hrir2hrtf(hrir_estimate, fs)
    if not np.array_equal(frequencies, estimate_frequencies):
        raise ValueError("Reference and estimate frequency grids differ")
    frequency_mask = (frequencies >= 20.0) & (frequencies <= 20000.0)
    lsd = 20.0 * np.log10(
        reference[:, :, frequency_mask] / estimate[:, :, frequency_mask]
    )
    matrix = np.sqrt(np.mean(np.square(lsd), axis=2))
    value = float(np.mean(matrix))
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite LAP2024 LSD")
    return value


def sam_ild_rms(hrir: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(hrir), axis=2))
    return 20.0 * np.log10(rms[:, 0]) - 20.0 * np.log10(rms[:, 1])


def lap2024_ild_mae(hrir_reference: np.ndarray, hrir_estimate: np.ndarray) -> float:
    value = float(np.mean(np.abs(sam_ild_rms(hrir_estimate) - sam_ild_rms(hrir_reference))))
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite LAP2024 ILD MAE")
    return value


def sam_itd_samples(hrir: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact SAM 0.0.8 MAXIACCe core, preserving its lag offset."""
    values = np.asarray(hrir)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("HRIR must have shape [location, ear, sample]")
    b, a = butter(10, 3000.0 / (fs / 2.0))
    itd_samples: list[int] = []
    maxiacc: list[float] = []
    for location in values:
        left = lfilter(b, a, location[0, :])
        right = lfilter(b, a, location[1, :])
        correlation = correlate(np.abs(hilbert(left)), np.abs(hilbert(right)))
        maxiacc.append(float(np.max(np.abs(correlation))))
        idx_lag = int(np.argmax(np.abs(correlation)))
        itd_samples.append(idx_lag - values.shape[2])
    return np.asarray(itd_samples, dtype=np.int64), np.asarray(maxiacc, dtype=np.float64)


def lap2024_itd_mae(hrir_reference: np.ndarray, hrir_estimate: np.ndarray, fs: float) -> float:
    reference_samples, _ = sam_itd_samples(hrir_reference, fs)
    estimate_samples, _ = sam_itd_samples(hrir_estimate, fs)
    value = float(np.mean(np.abs(estimate_samples - reference_samples)) / fs * 1e6)
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite LAP2024 ITD MAE")
    return value


def compatibility_test(wheel: Path, fs: float) -> dict[str, object]:
    official, temp_dir = load_official_sam(wheel)
    try:
        rng = np.random.default_rng(20260906)
        fixture = rng.normal(0.0, 0.01, size=(5, 2, 64)).astype(np.float64)
        fixture[:, 0, 10] += 1.0
        fixture[:, 1, 14] += 0.8
        estimate = fixture * 0.97

        official_lsd, official_lsd_matrix = official.calculate_lsd_across_locations(
            fixture, estimate, fs
        )
        our_lsd = lap2024_lsd(fixture, estimate, fs)
        official_ild = official.calculate_ild_difference(
            SimpleNamespace(hrir=fixture), SimpleNamespace(hrir=estimate)
        )
        our_ild = lap2024_ild_mae(fixture, estimate)

        source = textwrap.dedent(inspect.getsource(official.itd_estimator_maxiacce))
        source = source.replace("itd_s = itd_samps/fs", "itd_s = np.asarray(itd_samps)/fs")
        namespace = {"np": np, "sn": scipy.signal}
        exec(compile(source, "sam_v0.0.8_itd_estimator_type_shim", "exec"), namespace)
        official_itd, official_samples, official_maxiacc = namespace[
            "itd_estimator_maxiacce"
        ](fixture, fs)
        our_samples, our_maxiacc = sam_itd_samples(fixture, fs)
        our_itd = our_samples / fs
        official_itd_mae = float(np.mean(np.abs(official_itd - namespace["itd_estimator_maxiacce"](
            estimate, fs
        )[0])) * 1e6)
        our_itd_mae = lap2024_itd_mae(fixture, estimate, fs)

        if not np.isclose(official_lsd, our_lsd, rtol=0.0, atol=0.0):
            raise AssertionError("LSD mismatch against SAM 0.0.8")
        if not np.isclose(official_ild, our_ild, rtol=0.0, atol=0.0):
            raise AssertionError("ILD mismatch against SAM 0.0.8")
        if not np.array_equal(np.asarray(official_samples), our_samples):
            raise AssertionError("ITD sample lag mismatch against SAM 0.0.8")
        if not np.array_equal(np.asarray(official_maxiacc), our_maxiacc):
            raise AssertionError("ITD MAXIACCe mismatch against SAM 0.0.8")
        if not np.array_equal(np.asarray(official_itd), our_itd):
            raise AssertionError("ITD seconds mismatch against SAM 0.0.8")
        if not np.isclose(official_itd_mae, our_itd_mae, rtol=0.0, atol=0.0):
            raise AssertionError("ITD MAE mismatch against SAM 0.0.8")
        return {
            "status": "passed_with_official_v0.0.8_type_shim",
            "wheel_sha256": file_sha256(wheel),
            "source_python_hashes": wheel_members_sha256(wheel),
            "fixture_shape": list(fixture.shape),
            "fixture_sampling_rate_hz": fs,
            "lsd_abs_error": abs(float(official_lsd) - our_lsd),
            "ild_abs_error": abs(float(official_ild) - our_ild),
            "itd_samples_exact": True,
            "itd_maxiacc_exact": True,
            "itd_seconds_abs_error": float(np.max(np.abs(official_itd - our_itd))),
            "itd_mae_abs_error_us": abs(official_itd_mae - our_itd_mae),
            "official_runtime_issue": "SAM 0.0.8 executes list / fs and raises TypeError; np.asarray is the only applied type shim.",
        }
    finally:
        temp_dir.cleanup()


def load_source_manifest(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = manifest_identity(payload)
    if payload["status"] != "frozen" or payload["dataset"]["split"] != "test":
        raise PermissionError("Frozen test source manifest required")
    return payload, identity


def validate_inventory(root: Path, payload: Mapping[str, Any], labels: list[str]) -> None:
    dataset_root = root / payload["dataset"]["root"]
    source_paths = [(label, dataset_root / "subjects" / label / "q26.h5") for label in labels]
    if inventory_digest(source_paths) != payload["dataset"]["source_inventory_sha256"]:
        raise ValueError("Processed test-source inventory mismatch")
    sofa_root = root / payload["dataset"]["reference_sofa_root"]
    sofa_paths = [
        (label, sofa_root / (label + "_FreeFieldCompMinPhase_44kHz.sofa"))
        for label in labels
    ]
    if inventory_digest(sofa_paths) != payload["dataset"]["reference_sofa_inventory_sha256"]:
        raise ValueError("Reference SOFA inventory mismatch")
    for method, spec in payload["source_predictions"].items():
        prediction_root = root / spec["root"]
        paths = [
            (label, prediction_root / "subjects" / label / spec["filename"])
            for label in labels
        ]
        if inventory_digest(paths) != spec["inventory_sha256"]:
            raise ValueError(f"Frozen prediction inventory mismatch: {method}")


def get_residual_prediction(
    root: Path,
    source_payload: Mapping[str, Any],
    method: str,
    subject: str,
    mca_db: np.ndarray,
    metadata: Mapping[str, object],
) -> np.ndarray:
    spec = source_payload["source_predictions"][method]
    path = root / spec["root"] / "subjects" / subject / spec["filename"]
    _, hrir = load_residual_prediction(path, "test", mca_db, metadata)
    return hrir


def get_prediction_hrir(
    root: Path,
    source_payload: Mapping[str, Any],
    standardized_root: Path,
    method: str,
    subject: str,
    mca_db: np.ndarray,
    metadata: Mapping[str, object],
) -> np.ndarray:
    if method in STANDARDIZED_METHOD_PATHS:
        path = (
            standardized_root / "subjects" / subject
            / STANDARDIZED_METHOD_PATHS[method] / "prediction.h5"
        )
        _, hrir = load_standardized_prediction(path, "test")
        return hrir
    return get_residual_prediction(root, source_payload, method, subject, mca_db, metadata)


def main() -> None:
    args = parse_arguments()
    if not args.allow_test:
        raise PermissionError("Explicit --allow-test is required")
    root = ROOT
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    if config["status"] != "frozen" or config["split"] != "test" or not config["allow_test"]:
        raise PermissionError("Frozen LAP test config with allow_test=true required")
    if tuple(config["methods"]) != METHODS:
        raise ValueError("LAP method registry mismatch")

    fs = float(config["sampling_rate_hz"])
    compatibility = compatibility_test(args.sam_wheel.resolve(), fs)
    print(json.dumps({"sam_compatibility": compatibility["status"]}, ensure_ascii=False))
    if args.compatibility_only:
        print(json.dumps(compatibility, indent=2, ensure_ascii=False))
        return

    source_payload, source_identity = load_source_manifest(
        root, root / config["source_manifest"]
    )
    subject_rows = test_subject_paths(root / config["processed_test_root"], root / config["split_csv"])
    labels = [row[1] for row in subject_rows]
    if len(subject_rows) != 44 or len(set(labels)) != 44:
        raise ValueError("Expected exactly 44 unique test subjects")
    validate_inventory(root, source_payload, labels)
    standardized_root = root / config["standardized_prediction_root"]
    standardized_files = [
        standardized_root / "subjects" / label / directory / "prediction.h5"
        for label in labels
        for directory in STANDARDIZED_METHOD_PATHS.values()
    ]
    if len(standardized_files) != 352 or not all(path.is_file() for path in standardized_files):
        raise FileNotFoundError("Expected 352 standardized eight-method predictions")

    output_root = root / config["outputs"]["root"]
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    partial_root.mkdir(parents=True)
    (partial_root / config["outputs"]["compatibility"]).write_text(
        json.dumps(compatibility, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    values: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in METRIC_NAMES} for method in METHODS
    }
    sofa_root = root / config["reference_sofa_root"]
    dataset_root = root / config["processed_test_root"]

    for subject_index, (subject_id, subject_label, source_path) in enumerate(subject_rows, 1):
        with h5py.File(source_path, "r") as source:
            if decode_attribute(source.attrs["split"]) != "test":
                raise PermissionError("Non-test source encountered")
            mca_db = np.asarray(source["mca_logmag_db"][:], dtype=np.float32)
            directions = np.asarray(source["direction_features"][:], dtype=np.float32)
            interpolation = np.asarray(
                source["interpolation_evaluation_mask"][:], dtype=bool
            ).reshape(-1)
            q26_indices = np.asarray(
                source["sparse_direction_indices_zero_based"][:], dtype=np.int64
            ).reshape(-1)
            metadata = strict_metadata(source)
        if directions.shape != (793, 6) or mca_db.shape != (2, 793, 463):
            raise ValueError(f"Unexpected processed grid shape for {subject_label}")
        if q26_indices.size != 26 or np.count_nonzero(interpolation) != 767:
            raise ValueError(f"Q26/interpolation count mismatch for {subject_label}")
        q26_mask = np.zeros(793, dtype=bool)
        q26_mask[q26_indices] = True
        if not np.array_equal(~q26_mask, interpolation):
            raise ValueError(f"Q26/interpolation partition mismatch for {subject_label}")

        sofa_path = sofa_root / f"{subject_label}_FreeFieldCompMinPhase_44kHz.sofa"
        with h5py.File(sofa_path, "r") as sofa:
            reference_hrir = np.asarray(sofa["Data.IR"][:], dtype=np.float32)
            source_position = np.asarray(sofa["SourcePosition"][:], dtype=np.float64)
        if reference_hrir.shape != (793, 2, 256):
            raise ValueError(f"Reference HRIR shape mismatch for {subject_label}")
        if source_position.shape[0] != 793 or np.max(
            np.abs(source_position[:, :2] - directions[:, :2])
        ) > 1e-5:
            raise ValueError(f"Reference direction order mismatch for {subject_label}")

        for method in METHODS:
            predicted_hrir = get_prediction_hrir(
                root, source_payload, standardized_root, method, subject_label, mca_db, metadata
            )
            if predicted_hrir.shape != (793, 2, 256) or not np.all(np.isfinite(predicted_hrir)):
                raise ValueError(f"Invalid predicted HRIR for {subject_label}/{method}")
            metrics = {
                "LAP2024LSD_dB": lap2024_lsd(reference_hrir, predicted_hrir, fs),
                "LAP2024ILDMAE_dB": lap2024_ild_mae(reference_hrir, predicted_hrir),
                "LAP2024ITDMAE_us": lap2024_itd_mae(reference_hrir, predicted_hrir, fs),
            }
            for metric in METRIC_NAMES:
                value = metrics[metric]
                values[method][metric].append(value)
            rows.append({
                "SubjectLabel": subject_label,
                "SubjectID": subject_id,
                "MatchedDirectionCount": 793,
                "LAP2024LSD_dB": format(metrics["LAP2024LSD_dB"], ".17g"),
                "LAP2024ILDMAE_dB": format(metrics["LAP2024ILDMAE_dB"], ".17g"),
                "LAP2024ITDMAE_us": format(metrics["LAP2024ITDMAE_us"], ".17g"),
                "Method": method,
                "MethodLabel": source_payload["methods"][method]["label"],
            })
        print(f"LAP2024 test [{subject_index}/44] {subject_label}", flush=True)

    aggregate: list[dict[str, object]] = []
    for metric in METRIC_NAMES:
        means = {method: float(np.mean(values[method][metric])) for method in METHODS}
        for method in METHODS:
            vector = np.asarray(values[method][metric], dtype=np.float64)
            if vector.size != 44 or not np.all(np.isfinite(vector)):
                raise ValueError(f"Incomplete/nonfinite aggregate vector: {method}/{metric}")
            aggregate.append({
                "Method": method,
                "MethodLabel": source_payload["methods"][method]["label"],
                "Metric": metric,
                "SubjectCount": 44,
                "Mean": format(float(np.mean(vector)), ".17g"),
                "Std": format(float(np.std(vector, ddof=1)), ".17g"),
                "Median": format(float(np.median(vector)), ".17g"),
                "Min": format(float(np.min(vector)), ".17g"),
                "Max": format(float(np.max(vector)), ".17g"),
                "Rank": 1 + sum(means[other] < means[method] for other in METHODS),
                "Threshold": THRESHOLDS[metric],
                "BelowThreshold": bool(means[method] < THRESHOLDS[metric]),
            })

    if len(rows) != 44 * 10 or any(row["MatchedDirectionCount"] != 793 for row in rows):
        raise ValueError("Per-subject completeness check failed")
    write_csv(partial_root / config["outputs"]["per_subject"], rows)
    write_csv(partial_root / config["outputs"]["aggregate"], aggregate)
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "test",
        "allow_test": True,
        "test_subject_count_read": 44,
        "subject_count": 44,
        "method_count": 10,
        "methods": list(METHODS),
        "direction_count_per_subject": 793,
        "q26_direction_count": 26,
        "interpolation_direction_count": 767,
        "q26_included": True,
        "interpolation_mask_used_for_metrics": False,
        "solid_angle_weighting_used": False,
        "all_finite": True,
        "per_subject_rows": len(rows),
        "aggregate_rows": len(aggregate),
        "metrics": list(METRIC_NAMES),
        "thresholds": THRESHOLDS,
        "sam_compatibility": compatibility,
        "source_manifest_identity_sha256": source_identity,
        "strict_reconstruction_reused": True,
        "claim_boundary": "Locked test split descriptive evaluation; thresholds are descriptive and do not affect model selection.",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
    }
    (partial_root / config["outputs"]["summary"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial_root.replace(output_root)

    print("\nLAP 2024 Task 2 metrics (mean across 44 subjects; lower is better)")
    print("Method | LAP2024 LSD (dB) | LAP2024 ILD MAE (dB) | LAP2024 ITD MAE (us)")
    for method in METHODS:
        metric_rows = {row["Metric"]: row for row in aggregate if row["Method"] == method}
        print(
            f"{method} | {float(metric_rows['LAP2024LSD_dB']['Mean']):.6f} "
            f"(rank {metric_rows['LAP2024LSD_dB']['Rank']}) | "
            f"{float(metric_rows['LAP2024ILDMAE_dB']['Mean']):.6f} "
            f"(rank {metric_rows['LAP2024ILDMAE_dB']['Rank']}) | "
            f"{float(metric_rows['LAP2024ITDMAE_us']['Mean']):.6f} "
            f"(rank {metric_rows['LAP2024ITDMAE_us']['Rank']})"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
