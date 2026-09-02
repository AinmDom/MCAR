"""Evaluate frozen secondary/deferred endpoints for all ten methods on test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy
import torch

from mcar.evaluation.deferred_secondary_metrics import (
    dominant_notch_location_metrics,
    enable_external_torchaudio,
    strict_hrir_from_selected_db,
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
from mcar.training.train_film_siren_stage_c import horizontal_interpolation_indices


METHODS = (
    "SHOnly", "SUpDEqSH", "SUpDEqNN", "SUpDEqBary", "MCA",
    "MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED",
)
STANDARDIZED_METHOD_PATHS = {
    "SHOnly": "SHOnly",
    "SUpDEqSH": "SUpDEqSH",
    "SUpDEqNN": "SUpDEqNN",
    "SUpDEqBary": "SUpDEqBary",
    "MCA": "MCA",
    "MCARv351": "MCARv32",
    "FSPAE": "FSPAE",
    "RANF": "RANF",
}
SECONDARY_UNITS = {
    "FullSphereLSD": "dB",
    "HFFirstDifferenceMAE": "dB/bin",
    "HFSecondDifferenceMAE": "dB/bin^2",
    "MultiScaleNotchDepthMAE": "dB",
    "ERBBandILDMean": "dB",
    "SpatialLSD_0_10deg": "dB",
    "SpatialLSD_10_20deg": "dB",
    "SpatialLSD_20_30deg": "dB",
    "SpatialLSD_30_180deg": "dB",
}
DEFERRED_UNITS = {
    "DominantNotchPenalizedMAE_Hz": "Hz",
    "DominantNotchMatchedMAE_Hz": "Hz",
    "DominantNotchMissRate": "fraction",
    "DominantNotchSpuriousRate": "fraction",
    "ReferenceNotchFraction": "fraction",
    "ITDWeightedMAE_us": "us",
    "ITDMaximumAbsoluteError_us": "us",
}
UNITS = dict(SECONDARY_UNITS, **DEFERRED_UNITS)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def manifest_identity(payload: Mapping[str, Any]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256")).upper()
    actual = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Manifest identity mismatch")
    return actual


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def inventory_digest(paths: Sequence[Tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for key, path in sorted(paths, key=lambda item: item[0]):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest().upper()


def strict_metadata(handle: h5py.File) -> Dict[str, object]:
    group = handle["strict_ild"]
    return {
        "mca_selected_phase_rad": np.asarray(group["mca_selected_phase_rad"][:]),
        "mca_outside_real": np.asarray(group["mca_outside_real"][:]),
        "mca_outside_imag": np.asarray(group["mca_outside_imag"][:]),
        "selected_bin_indices_zero_based": np.asarray(
            group["selected_bin_indices_zero_based"][:]
        ),
        "outside_bin_indices_zero_based": np.asarray(
            group["outside_bin_indices_zero_based"][:]
        ),
        "single_sided_frequency_count": int(group.attrs["single_sided_frequency_count"]),
        "hrir_length": int(group.attrs["hrir_length"]),
    }


def itd_values(hrir: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimates = estimate_itd_seconds(
        torch.from_numpy(np.asarray(hrir, dtype=np.float32)).to(device),
        float(settings["sampling_rate_hz"]),
        upsampled_rate_hz=float(settings["upsampled_rate_hz"]),
        lowpass_hz=float(settings["lowpass_hz"]),
        maximum_itd_seconds=float(settings["maximum_itd_seconds"]),
        direction_batch_size=int(settings["direction_batch_size"]),
    )
    return estimates.cpu().numpy().astype(np.float64)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty table: {}".format(path.name))
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError("Inconsistent table columns: {}".format(path.name))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def derived_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2 ** 32)


def test_subject_paths(dataset_root: Path, split_csv: Path) -> List[Tuple[int, str, Path]]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split", "").lower() == "test"]
    if len(rows) != 44:
        raise ValueError("Expected 44 frozen test subjects")
    output = []
    for row in rows:
        label = row["subject_id"]
        if not label.startswith("P") or not label[1:].isdigit():
            raise ValueError("Invalid subject label: {}".format(label))
        path = dataset_root / "subjects" / label / "q26.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append((int(label[1:]), label, path))
    if len({item[0] for item in output}) != 44:
        raise ValueError("Duplicate test subject")
    return output


def load_standardized_prediction(path: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if decode_attribute(handle.attrs["split"]) != split:
            raise PermissionError("Prediction split mismatch: {}".format(path))
        magnitude_db = np.asarray(handle["predicted_magnitude_db"][:], dtype=np.float32)
        hrir = np.asarray(handle["predicted_hrir"][:], dtype=np.float32)
    if magnitude_db.shape != (2, 793, 463) or hrir.shape != (793, 2, 256):
        raise ValueError("Standardized prediction shape mismatch: {}".format(path))
    if not np.all(np.isfinite(magnitude_db)) or not np.all(np.isfinite(hrir)):
        raise FloatingPointError("Non-finite standardized prediction: {}".format(path))
    return magnitude_db, hrir


def load_residual_prediction(
    path: Path, split: str, mca_db: np.ndarray, metadata: Mapping[str, object]
) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if decode_attribute(handle.attrs["split"]) != split:
            raise PermissionError("Residual split mismatch: {}".format(path))
        residual = np.asarray(handle["predicted_residual_db"][:], dtype=np.float32)
    if residual.shape != (2, 793, 463) or not np.all(np.isfinite(residual)):
        raise FloatingPointError("Invalid residual prediction: {}".format(path))
    predicted_db = mca_db + residual
    hrir = strict_hrir_from_selected_db(predicted_db, metadata)
    return predicted_db, hrir.permute(1, 0, 2).cpu().numpy().astype(np.float32)


def verify_primary_reproduction(
    primary_path: Path, export_metric_path: Path, subject_labels: Sequence[str], tolerance: float
) -> float:
    accepted = set(STANDARDIZED_METHOD_PATHS)
    source: Dict[Tuple[str, str, str], float] = {}
    with primary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Method"] in accepted:
                source[(row["SubjectLabel"], row["Method"], row["Metric"])] = float(
                    row["Value_dB"]
                )
    observed: Dict[Tuple[str, str, str], float] = {}
    with export_metric_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = "MCARv351" if row["Method"] == "MCARv32" else row["Method"]
            if method in accepted:
                observed[(row["SubjectLabel"], method, row["Metric"])] = float(
                    row["Value_dB"]
                )
    expected_count = len(subject_labels) * len(accepted) * 4
    if len(source) != expected_count or set(source) != set(observed):
        raise ValueError("Primary reproduction keys are incomplete")
    maximum = max(abs(source[key] - observed[key]) for key in source)
    if maximum > tolerance:
        raise AssertionError("Primary reproduction failed: {:.17g}".format(maximum))
    return maximum


def main() -> None:
    args = parse_arguments()
    if not args.allow_test:
        raise PermissionError("Explicit --allow-test is required")
    root = project_root()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest_identity(payload)
    if payload["status"] != "frozen" or payload["dataset"]["split"] != "test":
        raise PermissionError("This evaluator requires the frozen test manifest")
    if not payload["authorization"]["authorized"]:
        raise PermissionError("Test authorization is not recorded")

    for resource in payload["implementation"]["resources"]:
        if file_sha256(root / resource["path"]) != resource["sha256"]:
            raise ValueError("Implementation hash mismatch: {}".format(resource["path"]))
    for resource in payload["implementation"]["external_resources"]:
        if file_sha256(Path(resource["path"])) != resource["sha256"]:
            raise ValueError("External dependency hash mismatch: {}".format(resource["path"]))
    expected_versions = payload["implementation"]["runtime_versions"]
    actual_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "scipy": scipy.__version__,
        "torch": str(torch.__version__),
    }
    if actual_versions != expected_versions:
        raise RuntimeError("Runtime mismatch: {} != {}".format(actual_versions, expected_versions))
    torchaudio_version = enable_external_torchaudio(
        Path(payload["implementation"]["torchaudio_site_packages"])
    )
    if torchaudio_version != payload["implementation"]["torchaudio_version"]:
        raise RuntimeError("torchaudio version mismatch")

    dataset_root = root / payload["dataset"]["root"]
    split_csv = root / payload["dataset"]["split_csv"]
    subject_rows = test_subject_paths(dataset_root, split_csv)
    subject_labels = [row[1] for row in subject_rows]
    if len(subject_rows) != 44 or len(set(subject_labels)) != 44:
        raise ValueError("Expected 44 unique test subjects")

    source_paths = [(label, path) for _, label, path in subject_rows]
    if inventory_digest(source_paths) != payload["dataset"]["source_inventory_sha256"]:
        raise ValueError("Processed test-source inventory mismatch")
    sofa_root = root / payload["dataset"]["reference_sofa_root"]
    sofa_paths = [
        (label, sofa_root / (label + "_FreeFieldCompMinPhase_44kHz.sofa"))
        for label in subject_labels
    ]
    if inventory_digest(sofa_paths) != payload["dataset"]["reference_sofa_inventory_sha256"]:
        raise ValueError("Reference SOFA inventory mismatch")

    for method, spec in payload["source_predictions"].items():
        prediction_root = root / spec["root"]
        filename = spec["filename"]
        paths = [(label, prediction_root / "subjects" / label / filename) for label in subject_labels]
        if inventory_digest(paths) != spec["inventory_sha256"]:
            raise ValueError("Frozen prediction inventory mismatch: {}".format(method))

    standardized_root = root / payload["standardized_export"]["prediction_root"]
    standardized_paths = []
    for label in subject_labels:
        for method, directory in STANDARDIZED_METHOD_PATHS.items():
            standardized_paths.append(
                (method + "/" + label, standardized_root / "subjects" / label / directory / "prediction.h5")
            )
    standardized_digest = inventory_digest(standardized_paths)
    primary_path = root / payload["primary_metrics"]["metric_long_csv"]
    export_metric_path = root / payload["standardized_export"]["metric_long_csv"]
    primary_max_error = verify_primary_reproduction(
        primary_path, export_metric_path, subject_labels,
        float(payload["standardized_export"]["primary_reproduction_tolerance_db"]),
    )

    output_root = root / payload["outputs"]["supplementary_results_root"]
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial_root.exists():
        raise FileExistsError("Refusing to overwrite supplementary test results")
    partial_root.mkdir(parents=True)

    per_subject_rows: List[Dict[str, object]] = []
    band_subject_rows: List[Dict[str, object]] = []
    direction_rows: List[Dict[str, object]] = []
    endpoint_values: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    band_values: Dict[str, List[np.ndarray]] = defaultdict(list)
    direction_values: Dict[str, List[np.ndarray]] = defaultdict(list)
    common_grid: Optional[Dict[str, np.ndarray]] = None
    itd_settings = payload["itd"]

    for subject_index, (subject_id, subject_label, source_path) in enumerate(subject_rows, 1):
        with h5py.File(source_path, "r") as source:
            if decode_attribute(source.attrs["split"]) != "test":
                raise PermissionError("Non-test source encountered")
            reference_db = np.asarray(source["reference_logmag_db"][:], dtype=np.float32)
            mca_db = np.asarray(source["mca_logmag_db"][:], dtype=np.float32)
            directions = np.asarray(source["direction_features"][:], dtype=np.float32)
            frequency = np.asarray(source["frequency_hz"][:], dtype=np.float32).reshape(-1)
            interpolation = np.asarray(
                source["interpolation_evaluation_mask"][:], dtype=bool
            ).reshape(-1)
            q26_indices = np.asarray(
                source["sparse_direction_indices_zero_based"][:], dtype=np.int64
            ).reshape(-1)
            metadata = strict_metadata(source)
        if reference_db.shape != (2, 793, 463) or mca_db.shape != reference_db.shape:
            raise ValueError("Source spectral shape mismatch")
        if directions.shape != (793, 6) or frequency.shape != (463,):
            raise ValueError("Source grid shape mismatch")
        q26_mask = np.zeros(793, dtype=bool)
        q26_mask[q26_indices] = True
        if q26_indices.size != 26 or np.count_nonzero(interpolation) != 767:
            raise ValueError("Q26/interpolation count mismatch")
        if not np.array_equal(~q26_mask, interpolation):
            raise ValueError("Q26/interpolation partition mismatch")
        horizontal = horizontal_interpolation_indices(directions, interpolation)
        if horizontal.size != 72:
            raise ValueError("Horizontal interpolation count mismatch")
        distance = minimum_q26_distance_degrees(directions[:, 2:5], q26_mask)
        if common_grid is None:
            common_grid = {
                "directions": directions.copy(), "frequency": frequency.copy(),
                "interpolation": interpolation.copy(), "distance": distance.copy(),
            }
        else:
            for key, value in (
                ("directions", directions), ("frequency", frequency),
                ("interpolation", interpolation), ("distance", distance),
            ):
                if not np.allclose(common_grid[key], value, rtol=0.0, atol=1e-6):
                    raise ValueError("Common grid mismatch: {}/{}".format(subject_label, key))

        sofa_path = sofa_root / (subject_label + "_FreeFieldCompMinPhase_44kHz.sofa")
        with h5py.File(sofa_path, "r") as sofa:
            reference_hrir = np.asarray(sofa["Data.IR"][:], dtype=np.float32)
            source_position = np.asarray(sofa["SourcePosition"][:], dtype=np.float64)
        if reference_hrir.shape != (793, 2, 256):
            raise ValueError("Reference HRIR shape mismatch")
        if np.max(np.abs(source_position[:, :2] - directions[:, :2])) > 1e-5:
            raise ValueError("Reference coordinate mismatch")
        reference_itd = itd_values(reference_hrir, itd_settings)
        interpolation_weights = normalized_direction_weights(directions[interpolation, 5])

        for method in METHODS:
            if method in STANDARDIZED_METHOD_PATHS:
                prediction_path = (
                    standardized_root / "subjects" / subject_label
                    / STANDARDIZED_METHOD_PATHS[method] / "prediction.h5"
                )
                predicted_db, predicted_hrir = load_standardized_prediction(
                    prediction_path, "test"
                )
            else:
                prediction_root = root / payload["source_predictions"][method]["root"]
                prediction_path = prediction_root / "subjects" / subject_label / "prediction.h5"
                predicted_db, predicted_hrir = load_residual_prediction(
                    prediction_path, "test", mca_db, metadata
                )

            lsd, direction_lsd, _ = full_sphere_lsd(
                predicted_db, reference_db, interpolation, directions[:, 5]
            )
            secondary = {"FullSphereLSD": lsd}
            secondary.update(
                spectral_shape_metrics(
                    predicted_db[:, interpolation, :],
                    reference_db[:, interpolation, :],
                    directions[interpolation, 5], frequency,
                )
            )
            band_centers, band_profile, band_mean = spectral_band_ild_profile(
                predicted_db[:, horizontal, :], reference_db[:, horizontal, :],
                directions[horizontal, 5], frequency,
            )
            secondary["ERBBandILDMean"] = band_mean
            for bin_label, value in distance_binned_lsd(
                direction_lsd, distance, interpolation, directions[:, 5]
            ).items():
                secondary["SpatialLSD_{}deg".format(bin_label)] = value

            notch = dominant_notch_location_metrics(
                predicted_db[:, interpolation, :], reference_db[:, interpolation, :],
                directions[interpolation, 5], frequency,
                matching_tolerance_hz=float(payload["notch"]["matching_tolerance_hz"]),
            )
            predicted_itd = itd_values(predicted_hrir, itd_settings)
            itd_error_us = np.abs(predicted_itd - reference_itd) * 1e6
            deferred = dict(notch)
            deferred["ITDWeightedMAE_us"] = float(
                np.sum(itd_error_us[interpolation] * interpolation_weights)
            )
            deferred["ITDMaximumAbsoluteError_us"] = float(
                np.max(itd_error_us[interpolation])
            )

            for tier, values in (("Secondary test", secondary), ("Deferred test", deferred)):
                for endpoint, value in values.items():
                    if endpoint not in UNITS or not np.isfinite(value):
                        raise FloatingPointError("Invalid endpoint {}/{}".format(method, endpoint))
                    endpoint_values[endpoint][method][subject_label] = float(value)
                    per_subject_rows.append({
                        "SubjectLabel": subject_label, "SubjectID": subject_id,
                        "Split": "test", "EvidenceTier": tier, "Method": method,
                        "MethodLabel": payload["methods"][method]["label"],
                        "Endpoint": endpoint, "Unit": UNITS[endpoint],
                        "Value": format(float(value), ".17g"),
                    })
            band_values[method].append(band_profile)
            direction_values[method].append(direction_lsd[interpolation])
            for band_index, (center, value) in enumerate(zip(band_centers, band_profile)):
                band_subject_rows.append({
                    "RecordType": "subject", "SubjectLabel": subject_label,
                    "SubjectID": subject_id, "Method": method, "BandIndex": band_index,
                    "BandCenter_Hz": format(float(center), ".17g"),
                    "ILDAbsoluteError_dB": format(float(value), ".17g"),
                })
        print("complete test metrics [{}/44] {}".format(subject_index, subject_label), flush=True)

    if set(endpoint_values) != set(UNITS):
        raise ValueError("Endpoint set mismatch")
    aggregate_rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []
    base_seed = int(payload["bootstrap"]["base_seed"])
    replicates = int(payload["bootstrap"]["replicates"])
    for endpoint in UNITS:
        if set(endpoint_values[endpoint]) != set(METHODS):
            raise ValueError("Incomplete method set for {}".format(endpoint))
        for method in METHODS:
            values = [endpoint_values[endpoint][method][label] for label in subject_labels]
            seed = derived_seed(base_seed, endpoint, method, "mean")
            mean, lower, upper = bootstrap_mean_interval(
                values, replicates=replicates, seed=seed
            )
            aggregate_rows.append({
                "EvidenceTier": "Secondary test" if endpoint in SECONDARY_UNITS else "Deferred test",
                "Method": method, "MethodLabel": payload["methods"][method]["label"],
                "Endpoint": endpoint, "Unit": UNITS[endpoint], "SubjectCount": 44,
                "Mean": format(mean, ".17g"),
                "SampleStd": format(float(np.std(values, ddof=1)), ".17g"),
                "Bootstrap95Lower": format(lower, ".17g"),
                "Bootstrap95Upper": format(upper, ".17g"), "BootstrapSeed": seed,
            })
        for baseline in METHODS[:-1]:
            seed = derived_seed(base_seed, endpoint, "BOUNDED", baseline, "paired")
            statistics = paired_tail_statistics(
                endpoint_values[endpoint]["BOUNDED"], endpoint_values[endpoint][baseline],
                replicates=replicates, seed=seed,
            )
            paired_rows.append({
                "EvidenceTier": "Secondary test" if endpoint in SECONDARY_UNITS else "Deferred test",
                "Endpoint": endpoint, "Unit": UNITS[endpoint], "Candidate": "BOUNDED",
                "Baseline": baseline, "Direction": "BOUNDED-minus-baseline; negative favors BOUNDED",
                "BootstrapSeed": seed, **statistics,
            })

    band_rows: List[Dict[str, object]] = []
    for method in METHODS:
        profiles = np.stack(band_values[method], axis=0)
        if profiles.shape != (44, 35):
            raise ValueError("Band profile shape mismatch")
        for band_index, center in enumerate(band_centers):
            seed = derived_seed(base_seed, "band", method, str(band_index))
            mean, lower, upper = bootstrap_mean_interval(
                profiles[:, band_index], replicates=replicates, seed=seed
            )
            band_rows.append({
                "Method": method, "BandIndex": band_index,
                "BandCenter_Hz": format(float(center), ".17g"), "SubjectCount": 44,
                "MeanILDAbsoluteError_dB": format(mean, ".17g"),
                "Bootstrap95Lower_dB": format(lower, ".17g"),
                "Bootstrap95Upper_dB": format(upper, ".17g"), "BootstrapSeed": seed,
            })

    assert common_grid is not None
    interpolation_indices = np.flatnonzero(common_grid["interpolation"].astype(bool))
    for method in METHODS:
        mean_map = np.stack(direction_values[method], axis=0).mean(axis=0)
        for local_index, direction_index in enumerate(interpolation_indices):
            features = common_grid["directions"][direction_index]
            direction_rows.append({
                "Method": method, "DirectionIndexZeroBased": int(direction_index),
                "Azimuth_deg": format(float(features[0]), ".17g"),
                "Elevation_deg": format(float(features[1]), ".17g"),
                "DistanceToQ26_deg": format(float(common_grid["distance"][direction_index]), ".17g"),
                "SolidAngleWeight": format(float(features[5]), ".17g"),
                "MeanLSD_dB": format(float(mean_map[local_index]), ".17g"),
            })

    expected_per_subject = 44 * len(METHODS) * len(UNITS)
    if len(per_subject_rows) != expected_per_subject:
        raise ValueError("Per-subject row count mismatch")
    write_csv(partial_root / "per_subject_metrics.csv", per_subject_rows)
    write_csv(partial_root / "aggregate_metrics.csv", aggregate_rows)
    write_csv(partial_root / "paired_vs_bounded.csv", paired_rows)
    write_csv(partial_root / "band_ild_profile.csv", band_rows)
    write_csv(partial_root / "spatial_direction_map.csv", direction_rows)
    quality = {
        "schema_version": "1.0", "status": "passed", "split": "test",
        "subject_count": 44, "test_subject_count_read": 44, "method_count": 10,
        "endpoint_count": 16, "per_subject_rows": len(per_subject_rows),
        "aggregate_rows": len(aggregate_rows), "paired_rows": len(paired_rows),
        "band_profile_rows": len(band_rows), "spatial_direction_rows": len(direction_rows),
        "direction_count": 793, "q26_direction_count": 26,
        "interpolation_direction_count": 767,
        "horizontal_interpolation_direction_count": 72, "frequency_bin_count": 463,
        "erb_band_count": 35, "all_finite": True,
        "primary_reproduction_max_abs_error_db": primary_max_error,
        "standardized_prediction_inventory_sha256": standardized_digest,
        "manifest_identity_sha256": identity,
    }
    (partial_root / "quality_checks.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    snapshot = dict(payload)
    snapshot["verified_identity_sha256"] = identity
    snapshot["standardized_prediction_inventory_sha256"] = standardized_digest
    (partial_root / "protocol_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "1.0", "status": "completed", "split": "test",
        "subject_count": 44, "test_subject_count_read": 44,
        "methods": list(METHODS), "secondary_endpoints": list(SECONDARY_UNITS),
        "deferred_endpoints": list(DEFERRED_UNITS), "all_finite": True,
        "bootstrap_replicates": replicates, "bootstrap_base_seed": base_seed,
        "manifest_identity_sha256": identity,
        "claim_boundary": "Post-lock supplementary test characterization; no tuning or selection.",
    }
    (partial_root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    partial_root.replace(output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
