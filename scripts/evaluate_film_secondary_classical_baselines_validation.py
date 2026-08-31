"""Extend frozen FiLM secondary metrics to five classical baselines on validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy
import torch

from mcar.evaluation.deferred_secondary_metrics import (
    dominant_notch_location_metrics,
    enable_external_torchaudio,
)
from mcar.evaluation.secondary_metrics import (
    bootstrap_mean_interval,
    distance_binned_lsd,
    full_sphere_lsd,
    minimum_q26_distance_degrees,
    normalized_direction_weights,
    paired_tail_statistics,
    spectral_band_ild_profile,
    spectral_shape_metrics,
)
from mcar.fsp_ae_signal import estimate_itd_seconds
from mcar.paths import project_root
from mcar.training.train_film_siren import split_subject_paths
from mcar.training.train_film_siren_stage_c import horizontal_interpolation_indices


METHODS = ("SHOnly", "SUpDEqSH", "SUpDEqNN", "SUpDEqBary", "MCA")
METHOD_LABELS = {
    "SHOnly": "SH only",
    "SUpDEqSH": "SUpDEq SH",
    "SUpDEqNN": "SUpDEq NN",
    "SUpDEqBary": "SUpDEq Barycentric",
    "MCA": "MCA",
}
SCALAR_UNITS = {
    "FullSphereLSD": "dB",
    "HFFirstDifferenceMAE": "dB/bin",
    "HFSecondDifferenceMAE": "dB/bin^2",
    "MultiScaleNotchDepthMAE": "dB",
    "ERBBandILDMean": "dB",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def manifest_identity(payload: dict[str, Any]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256")).upper()
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Manifest identity mismatch")
    return actual


def inventory_digest(root: Path, subjects: list[str], filename: str) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        source = root / "subjects" / subject / filename
        digest.update(subject.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def classical_inventory_digest(
    root: Path, subjects: list[str], method: str, filename: str
) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        source = root / "subjects" / subject / method / filename
        digest.update(subject.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def flat_inventory_digest(root: Path, subjects: list[str], filename_template: str) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        source = root / filename_template.format(subject=subject)
        digest.update(subject.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Inconsistent columns in {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def load_ranf_prediction(
    path: Path,
    selected_indices_zero_based: np.ndarray,
    directions: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load RANF SOFA HRIR and map its FFT to the frozen 463-bin grid."""
    with h5py.File(path, "r") as handle:
        hrir = np.asarray(handle["Data.IR"][:], dtype=np.float32)
        source_position = np.asarray(handle["SourcePosition"][:], dtype=np.float64)
        actual_rate = float(np.asarray(handle["Data.SamplingRate"][:]).reshape(-1)[0])
    if hrir.shape != (793, 2, 256):
        raise ValueError(f"Unexpected RANF HRIR shape: {path}")
    if source_position.shape != (793, 3):
        raise ValueError(f"Unexpected RANF coordinate shape: {path}")
    azimuth_error = np.abs(
        np.mod(source_position[:, 0] - directions[:, 0] + 180.0, 360.0) - 180.0
    )
    elevation_error = np.abs(source_position[:, 1] - directions[:, 1])
    if max(float(azimuth_error.max()), float(elevation_error.max())) > 1e-8:
        raise ValueError(f"RANF direction order mismatch: {path}")
    if abs(actual_rate - sampling_rate_hz) > 1e-9:
        raise ValueError(f"RANF sampling-rate mismatch: {path}")
    spectrum = np.fft.fft(hrir.astype(np.float64), n=1024, axis=-1)
    selected = spectrum[..., selected_indices_zero_based]
    selected_db = 20.0 * np.log10(np.maximum(np.abs(selected), 1e-10))
    return np.transpose(selected_db, (1, 0, 2)), hrir


def load_fspae_prediction(
    path: Path,
    selected_indices_zero_based: np.ndarray,
    source_frequency_hz: np.ndarray,
    subject_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load FSP-AE HRIR and apply its preregistered one-bin frequency offset."""
    with h5py.File(path, "r") as handle:
        if decode_attribute(handle.attrs["split"]) != "val":
            raise PermissionError(f"Non-validation FSP-AE input: {path}")
        if decode_attribute(handle.attrs["subject_id"]) != subject_label:
            raise ValueError(f"FSP-AE subject mismatch: {path}")
        magnitude_db = np.asarray(handle["predicted_magnitude_db"][:], dtype=np.float32)
        hrir = np.asarray(handle["predicted_hrir"][:], dtype=np.float32)
        frequency_hz = np.asarray(handle["frequency_hz"][:], dtype=np.float64).reshape(-1)
    if magnitude_db.shape != (793, 2, 512) or hrir.shape != (793, 2, 256):
        raise ValueError(f"Unexpected FSP-AE prediction shape: {path}")
    if np.any(selected_indices_zero_based < 1):
        raise ValueError("FSP-AE mapping requires source selected bins to exclude DC")
    fsp_indices = selected_indices_zero_based - 1
    if not np.allclose(
        frequency_hz[fsp_indices], source_frequency_hz, rtol=0.0, atol=1e-4
    ):
        raise ValueError(f"FSP-AE/source frequency mapping mismatch: {path}")
    selected_db = magnitude_db[..., fsp_indices]
    return np.transpose(selected_db, (1, 0, 2)), hrir


def load_classical_prediction(
    path: Path,
    source_frequency_hz: np.ndarray,
    subject_label: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if decode_attribute(handle.attrs["split"]) != "val":
            raise PermissionError(f"Non-validation classical input: {path}")
        if decode_attribute(handle.attrs["subject_id"]) != subject_label:
            raise ValueError(f"Classical subject mismatch: {path}")
        if decode_attribute(handle.attrs["method"]) != method:
            raise ValueError(f"Classical method mismatch: {path}")
        magnitude_db = np.asarray(handle["predicted_magnitude_db"][:], dtype=np.float32)
        hrir = np.asarray(handle["predicted_hrir"][:], dtype=np.float32)
        frequency_hz = np.asarray(handle["frequency_hz"][:], dtype=np.float64).reshape(-1)
    if magnitude_db.shape != (2, 793, 463) or hrir.shape != (793, 2, 256):
        raise ValueError(f"Unexpected classical prediction shape: {path}")
    if not np.allclose(frequency_hz, source_frequency_hz, rtol=0.0, atol=1e-8):
        raise ValueError(f"Classical frequency grid mismatch: {path}")
    return magnitude_db, hrir


def itd_values(hrir: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(hrir, dtype=np.float32)).cuda()
    estimates = estimate_itd_seconds(
        tensor,
        float(settings["sampling_rate_hz"]),
        upsampled_rate_hz=float(settings["upsampled_rate_hz"]),
        lowpass_hz=float(settings["lowpass_hz"]),
        maximum_itd_seconds=float(settings["maximum_itd_seconds"]),
        direction_batch_size=int(settings["direction_batch_size"]),
    )
    return estimates.cpu().numpy().astype(np.float64)


def load_metric_rows(
    path: Path,
    *,
    method_column: str,
    value_column: str,
    endpoint_column: str,
    accepted_methods: dict[str, str],
    record_type: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if record_type is not None and row.get("RecordType") != record_type:
                continue
            source_method = row[method_column]
            if source_method not in accepted_methods:
                continue
            endpoint = row[endpoint_column]
            method = accepted_methods[source_method]
            output[endpoint][method][row["SubjectLabel"]] = float(row[value_column])
    return output


def add_nested_values(
    target: dict[str, dict[str, dict[str, float]]],
    source: dict[str, dict[str, dict[str, float]]],
) -> None:
    for endpoint, methods in source.items():
        for method, subjects in methods.items():
            if target[endpoint][method]:
                raise ValueError(f"Duplicate values for {endpoint}/{method}")
            target[endpoint][method].update(subjects)


def main() -> None:
    args = parse_arguments()
    root = project_root()
    payload = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    identity = manifest_identity(payload)
    if payload["status"] != "frozen" or payload["dataset"]["split"] != "val":
        raise PermissionError("Baseline extension is frozen to validation only")
    if int(payload["dataset"]["subject_count"]) != 44:
        raise ValueError("Expected 44 validation subjects")

    for resource in payload["implementation"]["resources"]:
        if file_sha256(root / resource["path"]) != resource["sha256"]:
            raise ValueError(f"Implementation hash mismatch: {resource['path']}")
    for resource in payload["implementation"]["external_resources"]:
        if file_sha256(Path(resource["path"])) != resource["sha256"]:
            raise ValueError(f"External dependency hash mismatch: {resource['path']}")
    expected_versions = payload["implementation"]["runtime_versions"]
    actual_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "scipy": scipy.__version__,
        "torch": str(torch.__version__),
    }
    if actual_versions != expected_versions:
        raise RuntimeError(
            f"Runtime version mismatch: expected {expected_versions}, found {actual_versions}"
        )
    torchaudio_version = enable_external_torchaudio(
        Path(payload["implementation"]["torchaudio_site_packages"])
    )
    if torchaudio_version != payload["implementation"]["torchaudio_version"]:
        raise RuntimeError("torchaudio version mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen ITD estimator")

    dataset_root = (root / payload["dataset"]["root"]).resolve()
    split_csv = (root / payload["dataset"]["split_csv"]).resolve()
    for resource in payload["dataset"]["resources"]:
        if file_sha256(root / resource["path"]) != resource["sha256"]:
            raise ValueError(f"Dataset resource hash mismatch: {resource['path']}")
    subject_rows = split_subject_paths(dataset_root, split_csv, "val")
    subject_labels = [row[1] for row in subject_rows]
    if len(subject_labels) != 44 or len(set(subject_labels)) != 44:
        raise ValueError("Validation subject identities are incomplete")
    sofa_root = root / payload["dataset"]["reference_sofa_root"]
    if flat_inventory_digest(
        sofa_root, subject_labels, payload["dataset"]["reference_sofa_filename_template"]
    ) != payload["dataset"]["reference_sofa_inventory_sha256"]:
        raise ValueError("Reference SOFA inventory mismatch")

    method_specs = payload["methods"]
    if tuple(method_specs) != METHODS:
        raise ValueError("Comparator order changed")
    method_roots: dict[str, Path] = {}
    for method, spec in method_specs.items():
        prediction_root = root / spec["prediction_root"]
        if classical_inventory_digest(
            prediction_root, subject_labels, method, spec["filename"]
        ) != spec["prediction_inventory_sha256"]:
            raise ValueError(f"Prediction inventory mismatch: {method}")
        if file_sha256(root / spec["provenance_path"]) != spec["provenance_sha256"]:
            raise ValueError(f"Provenance hash mismatch: {method}")
        method_roots[method] = prediction_root

    for source in payload["existing_results"].values():
        if file_sha256(root / source["path"]) != source["sha256"]:
            raise ValueError(f"Existing-result hash mismatch: {source['path']}")

    output_root = root / payload["output_root"]
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial_root.exists():
        raise FileExistsError("Refusing to overwrite baseline-extension output")
    partial_root.mkdir(parents=True)

    per_subject_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    spatial_map_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    scalar_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    distance_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    deferred_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    band_values: dict[str, list[np.ndarray]] = defaultdict(list)
    direction_values: dict[str, list[np.ndarray]] = defaultdict(list)
    common_grid: dict[str, np.ndarray] | None = None
    max_prediction_frequency_error_hz = 0.0
    itd_settings = payload["itd"]

    for subject_index, (subject_id, subject_label, source_path) in enumerate(
        subject_rows, start=1
    ):
        with h5py.File(source_path, "r") as source:
            if decode_attribute(source.attrs["split"]) != "val":
                raise PermissionError(f"Non-validation source: {source_path}")
            reference_db = np.asarray(source["reference_logmag_db"][:], dtype=np.float32)
            directions = np.asarray(source["direction_features"][:], dtype=np.float32)
            frequency = np.asarray(source["frequency_hz"][:], dtype=np.float64).reshape(-1)
            interpolation = np.asarray(
                source["interpolation_evaluation_mask"][:], dtype=bool
            ).reshape(-1)
            q26_indices = np.asarray(
                source["sparse_direction_indices_zero_based"][:], dtype=np.int64
            ).reshape(-1)
            selected_indices = np.asarray(
                source["strict_ild/selected_bin_indices_zero_based"][:], dtype=np.int64
            ).reshape(-1)
            sampling_rate = float(np.asarray(source.attrs["sampling_rate_hz"]).item())
        if reference_db.shape != (2, 793, 463):
            raise ValueError(f"Reference spectral shape mismatch: {subject_label}")
        if directions.shape != (793, 6) or frequency.shape != (463,):
            raise ValueError(f"Common-grid shape mismatch: {subject_label}")
        if selected_indices.shape != (463,) or np.count_nonzero(interpolation) != 767:
            raise ValueError(f"Mask/frequency selection mismatch: {subject_label}")
        q26_mask = np.zeros(793, dtype=bool)
        q26_mask[q26_indices] = True
        if not np.array_equal(~q26_mask, interpolation):
            raise ValueError(f"Q26/interpolation partition mismatch: {subject_label}")
        horizontal = horizontal_interpolation_indices(directions, interpolation)
        distance = minimum_q26_distance_degrees(directions[:, 2:5], q26_mask)
        if common_grid is None:
            common_grid = {
                "directions": directions.copy(),
                "frequency": frequency.copy(),
                "interpolation": interpolation.copy(),
                "distance": distance.copy(),
            }
        else:
            for key, value in (
                ("directions", directions),
                ("frequency", frequency),
                ("interpolation", interpolation),
                ("distance", distance),
            ):
                if not np.allclose(common_grid[key], value, rtol=0.0, atol=1e-6):
                    raise ValueError(f"Common grid mismatch: {subject_label}/{key}")

        reference_sofa = sofa_root / payload["dataset"][
            "reference_sofa_filename_template"
        ].format(subject=subject_label)
        with h5py.File(reference_sofa, "r") as handle:
            reference_hrir = np.asarray(handle["Data.IR"][:], dtype=np.float32)
            source_position = np.asarray(handle["SourcePosition"][:], dtype=np.float64)
        if reference_hrir.shape != (793, 2, 256):
            raise ValueError(f"Reference HRIR shape mismatch: {subject_label}")
        reference_coordinate_error = float(
            np.max(np.abs(source_position[:, :2] - directions[:, :2]))
        )
        if reference_coordinate_error > float(
            payload["dataset"]["coordinate_tolerance_degrees"]
        ):
            raise ValueError(f"Reference direction mismatch: {subject_label}")
        reference_itd = itd_values(reference_hrir, itd_settings)

        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in METHODS:
            prediction_path = (
                method_roots[method]
                / "subjects"
                / subject_label
                / method
                / method_specs[method]["filename"]
            )
            predictions[method] = load_classical_prediction(
                prediction_path, frequency, subject_label, method
            )

        direction_weights = directions[:, 5]
        interpolation_weights = normalized_direction_weights(
            direction_weights[interpolation]
        )
        for method in METHODS:
            predicted_db, predicted_hrir = predictions[method]
            if predicted_db.shape != reference_db.shape or not np.all(
                np.isfinite(predicted_db)
            ):
                raise FloatingPointError(f"Invalid spectra: {method}/{subject_label}")
            if not np.all(np.isfinite(predicted_hrir)):
                raise FloatingPointError(f"Invalid HRIR: {method}/{subject_label}")
            lsd, direction_lsd, _ = full_sphere_lsd(
                predicted_db, reference_db, interpolation, direction_weights
            )
            metrics = {"FullSphereLSD": lsd}
            metrics.update(
                spectral_shape_metrics(
                    predicted_db[:, interpolation, :],
                    reference_db[:, interpolation, :],
                    direction_weights[interpolation],
                    frequency,
                )
            )
            band_centers, profile, band_mean = spectral_band_ild_profile(
                predicted_db[:, horizontal, :],
                reference_db[:, horizontal, :],
                direction_weights[horizontal],
                frequency,
            )
            metrics["ERBBandILDMean"] = band_mean
            for endpoint, value in metrics.items():
                scalar_values[endpoint][method][subject_label] = value
                per_subject_rows.append(
                    {
                        "SubjectLabel": subject_label,
                        "SubjectID": subject_id,
                        "Split": "val",
                        "Method": method,
                        "MethodLabel": METHOD_LABELS[method],
                        "Endpoint": endpoint,
                        "Unit": SCALAR_UNITS[endpoint],
                        "Value": format(value, ".17g"),
                    }
                )
            band_values[method].append(profile)
            for band_index, (center, value) in enumerate(zip(band_centers, profile)):
                band_rows.append(
                    {
                        "RecordType": "subject",
                        "SubjectLabel": subject_label,
                        "SubjectID": subject_id,
                        "Method": method,
                        "BandIndex": band_index,
                        "BandCenter_Hz": format(float(center), ".17g"),
                        "ILDAbsoluteError_dB": format(float(value), ".17g"),
                        "Bootstrap95Lower_dB": "",
                        "Bootstrap95Upper_dB": "",
                    }
                )
            bins = distance_binned_lsd(
                direction_lsd, distance, interpolation, direction_weights
            )
            for bin_label, value in bins.items():
                endpoint = f"SpatialLSD_{bin_label}deg"
                distance_values[endpoint][method][subject_label] = value
                spatial_rows.append(
                    {
                        "RecordType": "subject",
                        "SubjectLabel": subject_label,
                        "SubjectID": subject_id,
                        "Method": method,
                        "DistanceBin_deg": bin_label,
                        "LSD_dB": format(value, ".17g"),
                        "Bootstrap95Lower_dB": "",
                        "Bootstrap95Upper_dB": "",
                    }
                )
            direction_values[method].append(direction_lsd[interpolation])

            notch = dominant_notch_location_metrics(
                predicted_db[:, interpolation, :],
                reference_db[:, interpolation, :],
                direction_weights[interpolation],
                frequency,
                matching_tolerance_hz=float(payload["notch"]["matching_tolerance_hz"]),
            )
            predicted_itd = itd_values(predicted_hrir, itd_settings)
            itd_error_us = np.abs(predicted_itd - reference_itd) * 1e6
            deferred = {
                **notch,
                "ITDWeightedMAE_us": float(
                    np.sum(itd_error_us[interpolation] * interpolation_weights)
                ),
                "ITDMaximumAbsoluteError_us": float(
                    np.max(itd_error_us[interpolation])
                ),
            }
            for endpoint, value in deferred.items():
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite {endpoint}: {method}/{subject_label}"
                    )
                deferred_values[endpoint][method][subject_label] = value
                endpoint_rows.append(
                    {
                        "SubjectLabel": subject_label,
                        "SubjectID": subject_id,
                        "Method": method,
                        "Endpoint": endpoint,
                        "Value": format(value, ".17g"),
                    }
                )
        print(f"baseline extension [{subject_index}/44] {subject_label}", flush=True)

    assert common_grid is not None
    aggregate_rows: list[dict[str, object]] = []
    for endpoint, methods in scalar_values.items():
        for method in METHODS:
            values = [methods[method][label] for label in subject_labels]
            mean, lower, upper = bootstrap_mean_interval(values)
            aggregate_rows.append(
                {
                    "Method": method,
                    "MethodLabel": METHOD_LABELS[method],
                    "Endpoint": endpoint,
                    "Unit": SCALAR_UNITS[endpoint],
                    "SubjectCount": 44,
                    "Mean": format(mean, ".17g"),
                    "SampleStd": format(float(np.std(values, ddof=1)), ".17g"),
                    "Bootstrap95Lower": format(lower, ".17g"),
                    "Bootstrap95Upper": format(upper, ".17g"),
                }
            )

    for method in METHODS:
        profiles = np.stack(band_values[method], axis=0)
        for band_index, center in enumerate(band_centers):
            mean, lower, upper = bootstrap_mean_interval(profiles[:, band_index])
            band_rows.append(
                {
                    "RecordType": "aggregate",
                    "SubjectLabel": "",
                    "SubjectID": "",
                    "Method": method,
                    "BandIndex": band_index,
                    "BandCenter_Hz": format(float(center), ".17g"),
                    "ILDAbsoluteError_dB": format(mean, ".17g"),
                    "Bootstrap95Lower_dB": format(lower, ".17g"),
                    "Bootstrap95Upper_dB": format(upper, ".17g"),
                }
            )
    for endpoint, methods in distance_values.items():
        bin_label = endpoint.removeprefix("SpatialLSD_").removesuffix("deg")
        for method in METHODS:
            values = [methods[method][label] for label in subject_labels]
            mean, lower, upper = bootstrap_mean_interval(values)
            spatial_rows.append(
                {
                    "RecordType": "aggregate",
                    "SubjectLabel": "",
                    "SubjectID": "",
                    "Method": method,
                    "DistanceBin_deg": bin_label,
                    "LSD_dB": format(mean, ".17g"),
                    "Bootstrap95Lower_dB": format(lower, ".17g"),
                    "Bootstrap95Upper_dB": format(upper, ".17g"),
                }
            )
    interpolation_indices = np.flatnonzero(common_grid["interpolation"])
    for method in METHODS:
        mean_map = np.stack(direction_values[method], axis=0).mean(axis=0)
        for local_index, direction_index in enumerate(interpolation_indices):
            features = common_grid["directions"][direction_index]
            spatial_map_rows.append(
                {
                    "Method": method,
                    "DirectionIndexZeroBased": int(direction_index),
                    "Azimuth_deg": format(float(features[0]), ".17g"),
                    "Elevation_deg": format(float(features[1]), ".17g"),
                    "DistanceToQ26_deg": format(
                        float(common_grid["distance"][direction_index]), ".17g"
                    ),
                    "SolidAngleWeight": format(float(features[5]), ".17g"),
                    "MeanLSD_dB": format(float(mean_map[local_index]), ".17g"),
                }
            )

    existing = payload["existing_results"]
    candidate_secondary = load_metric_rows(
        root / existing["bounded_secondary"]["path"],
        method_column="Method",
        value_column="Value",
        endpoint_column="Endpoint",
        accepted_methods={"BOUNDED": "BOUNDED"},
    )
    candidate_spatial_raw = load_metric_rows(
        root / existing["bounded_spatial"]["path"],
        method_column="Method",
        value_column="LSD_dB",
        endpoint_column="DistanceBin_deg",
        accepted_methods={"BOUNDED": "BOUNDED"},
        record_type="subject",
    )
    candidate_spatial: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for bin_label, methods in candidate_spatial_raw.items():
        candidate_spatial[f"SpatialLSD_{bin_label}deg"] = methods
    candidate_primary = load_metric_rows(
        root / existing["bounded_primary"]["path"],
        method_column="Method",
        value_column="Value_dB",
        endpoint_column="Metric",
        accepted_methods={"HYBRID": "BOUNDED"},
    )
    comparator_primary = load_metric_rows(
        root / existing["comparator_primary"]["path"],
        method_column="Method",
        value_column="Value_dB",
        endpoint_column="Metric",
        accepted_methods={method: method for method in METHODS},
    )
    tail_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for source in (
        candidate_secondary,
        candidate_spatial,
        candidate_primary,
        scalar_values,
        distance_values,
        comparator_primary,
    ):
        add_nested_values(tail_values, source)
    tail_rows: list[dict[str, object]] = []
    for endpoint, methods in tail_values.items():
        if set(methods) != {"BOUNDED", *METHODS}:
            raise ValueError(f"Incomplete paired endpoint: {endpoint}/{sorted(methods)}")
        unit = "dB" if endpoint in comparator_primary else SCALAR_UNITS.get(endpoint, "dB")
        for baseline in METHODS:
            tail_rows.append(
                {
                    "Endpoint": endpoint,
                    "Unit": unit,
                    "Baseline": baseline,
                    "Difference": "BOUNDED-minus-baseline; negative favors BOUNDED",
                    **paired_tail_statistics(methods["BOUNDED"], methods[baseline]),
                }
            )

    deferred_aggregate_rows: list[dict[str, object]] = []
    for endpoint, methods in deferred_values.items():
        for method in METHODS:
            values = [methods[method][label] for label in subject_labels]
            mean, lower, upper = bootstrap_mean_interval(values)
            deferred_aggregate_rows.append(
                {
                    "Method": method,
                    "Endpoint": endpoint,
                    "SubjectCount": 44,
                    "Mean": format(mean, ".17g"),
                    "SampleStd": format(float(np.std(values, ddof=1)), ".17g"),
                    "Bootstrap95Lower": format(lower, ".17g"),
                    "Bootstrap95Upper": format(upper, ".17g"),
                }
            )
    candidate_deferred = load_metric_rows(
        root / existing["bounded_deferred"]["path"],
        method_column="Method",
        value_column="Value",
        endpoint_column="Endpoint",
        accepted_methods={"BOUNDED": "BOUNDED"},
    )
    deferred_paired_rows: list[dict[str, object]] = []
    for endpoint, methods in deferred_values.items():
        if not (endpoint.startswith("ITD") or endpoint.startswith("DominantNotch")):
            continue
        for baseline in METHODS:
            deferred_paired_rows.append(
                {
                    "Endpoint": endpoint,
                    "Baseline": baseline,
                    **paired_tail_statistics(
                        candidate_deferred[endpoint]["BOUNDED"], methods[baseline]
                    ),
                }
            )

    write_csv(partial_root / "per_subject_metrics.csv", per_subject_rows)
    write_csv(partial_root / "aggregate_metrics.csv", aggregate_rows)
    write_csv(partial_root / "paired_tail_risk.csv", tail_rows)
    write_csv(partial_root / "band_ild_profile.csv", band_rows)
    write_csv(partial_root / "spatial_distance_bins.csv", spatial_rows)
    write_csv(partial_root / "spatial_direction_map.csv", spatial_map_rows)
    write_csv(partial_root / "per_subject_endpoints.csv", endpoint_rows)
    write_csv(partial_root / "aggregate_endpoints.csv", deferred_aggregate_rows)
    write_csv(partial_root / "paired_deferred_tail_risk.csv", deferred_paired_rows)
    quality = {
        "status": "passed",
        "split": "val",
        "subject_count": 44,
        "method_count": 5,
        "methods": list(METHODS),
        "test_subject_count_read": 0,
        "all_finite": True,
        "direction_count": 793,
        "interpolation_direction_count": 767,
        "horizontal_interpolation_direction_count": 72,
        "frequency_bin_count": 463,
        "erb_band_count": int(len(band_centers)),
        "max_prediction_frequency_error_hz": max_prediction_frequency_error_hz,
        "row_counts": {
            "per_subject_metrics": len(per_subject_rows),
            "aggregate_metrics": len(aggregate_rows),
            "paired_tail_risk": len(tail_rows),
            "band_ild_profile": len(band_rows),
            "spatial_distance_bins": len(spatial_rows),
            "spatial_direction_map": len(spatial_map_rows),
            "per_subject_endpoints": len(endpoint_rows),
            "aggregate_endpoints": len(deferred_aggregate_rows),
            "paired_deferred_tail_risk": len(deferred_paired_rows),
        },
        "mechanism_comparison_status": "not_applicable_no_common_internal_quantity",
        "efficiency_comparison_status": "not_available_no_same_hardware_benchmark",
        "localization_status": "not_run_dependency_unavailable",
        "manifest_identity_sha256": identity,
    }
    (partial_root / "quality_checks.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "completed",
        "split": "val",
        "subject_count": 44,
        "methods": list(METHODS),
        "test_subject_count_read": 0,
        "secondary_endpoints": list(SCALAR_UNITS),
        "deferred_endpoints": sorted(deferred_values),
        "paired_baselines": list(METHODS),
        "manifest_identity_sha256": identity,
        "mechanism_comparison_status": quality["mechanism_comparison_status"],
        "efficiency_comparison_status": quality["efficiency_comparison_status"],
        "localization_status": quality["localization_status"],
    }
    (partial_root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (partial_root / "protocol_snapshot.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    partial_root.replace(output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
