"""Evaluate the result-blind FiLM secondary-metrics tranche on validation only."""

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
import torch

from mcar.evaluation.secondary_metrics import (
    DISTANCE_BIN_LABELS,
    bootstrap_mean_interval,
    distance_binned_lsd,
    full_sphere_lsd,
    minimum_q26_distance_degrees,
    paired_tail_statistics,
    spectral_band_ild_profile,
    spectral_shape_metrics,
)
from mcar.paths import project_root
from mcar.training.train_film_siren_stage_c import horizontal_interpolation_indices
from mcar.training.train_film_siren import split_subject_paths


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


def prediction_inventory_digest(root: Path, subjects: list[str]) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        path = root / "subjects" / subject / "prediction.h5"
        digest.update(subject.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path.name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Inconsistent columns for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_primary_metrics(
    path: Path, method_mapping: dict[str, str]
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_method = row["Method"]
            if source_method not in method_mapping:
                continue
            endpoint = row["Metric"]
            method = method_mapping[source_method]
            output[endpoint][method][row["SubjectLabel"]] = float(row["Value_dB"])
    return output


def main() -> None:
    arguments = parse_arguments()
    root = project_root()
    manifest_path = arguments.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest_identity(payload)
    if payload["status"] != "frozen" or payload["dataset"]["split"] != "val":
        raise PermissionError("Secondary evaluation is frozen to validation only")
    if int(payload["dataset"]["subject_count"]) != 44:
        raise ValueError("Expected a frozen 44-subject validation manifest")

    for resource in payload["implementation"]["resources"]:
        path = root / resource["path"]
        if file_sha256(path) != resource["sha256"]:
            raise ValueError(f"Implementation hash mismatch: {path}")
    versions = payload["implementation"]["runtime_versions"]
    actual_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "torch": str(torch.__version__),
    }
    if actual_versions != versions:
        raise RuntimeError(
            f"Runtime version mismatch: expected {versions}, found {actual_versions}"
        )

    dataset_root = (root / payload["dataset"]["root"]).resolve()
    split_csv = (root / payload["dataset"]["split_csv"]).resolve()
    for resource in payload["dataset"]["resources"]:
        path = root / resource["path"]
        if file_sha256(path) != resource["sha256"]:
            raise ValueError(f"Dataset resource hash mismatch: {path}")
    if np.random.default_rng(0).bit_generator.__class__.__name__ != "PCG64":
        raise RuntimeError("The frozen bootstrap generator is NumPy PCG64")
    subject_rows = split_subject_paths(dataset_root, split_csv, "val")
    subject_labels = [row[1] for row in subject_rows]
    if len(subject_labels) != 44 or len(set(subject_labels)) != 44:
        raise ValueError("Validation subject identities are incomplete")

    method_specs = payload["methods"]
    if list(method_specs) != ["BOUNDED", "HYBRID", "FILMENS", "MCAR"]:
        raise ValueError("Method order or comparator set changed")
    prediction_roots: dict[str, Path] = {}
    for method, spec in method_specs.items():
        prediction_root = (root / spec["prediction_root"]).resolve()
        report = root / spec["inference_report"]
        if file_sha256(report) != spec["inference_report_sha256"]:
            raise ValueError(f"Inference report hash mismatch for {method}")
        actual_inventory = prediction_inventory_digest(prediction_root, subject_labels)
        if actual_inventory != spec["prediction_inventory_sha256"]:
            raise ValueError(f"Prediction inventory hash mismatch for {method}")
        prediction_roots[method] = prediction_root

    primary_path = root / payload["primary_metrics"]["metric_long_csv"]
    if file_sha256(primary_path) != payload["primary_metrics"]["sha256"]:
        raise ValueError("Primary metric source hash mismatch")

    output_root = (root / payload["output_root"]).resolve()
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root} or {partial_root}")
    partial_root.mkdir(parents=True)

    per_subject_rows: list[dict[str, object]] = []
    band_subject_rows: list[dict[str, object]] = []
    spatial_subject_rows: list[dict[str, object]] = []
    direction_subject_rows: list[dict[str, object]] = []
    scalar_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    band_values: dict[str, list[np.ndarray]] = defaultdict(list)
    direction_values: dict[str, list[np.ndarray]] = defaultdict(list)
    distance_by_subject: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    common_grid: dict[str, np.ndarray] | None = None
    method_labels = {method: spec["label"] for method, spec in method_specs.items()}
    bootstrap_replicates = int(payload["registered_tranche"]["bootstrap_replicates"])
    bootstrap_seed = int(payload["registered_tranche"]["bootstrap_seed"])
    tie_tolerance = float(payload["registered_tranche"]["tie_tolerance"])
    units = {
        "FullSphereLSD": "dB",
        "HFFirstDifferenceMAE": "dB/bin",
        "HFSecondDifferenceMAE": "dB/bin^2",
        "MultiScaleNotchDepthMAE": "dB",
        "ERBBandILDMean": "dB",
    }

    for subject_index, (subject_id, subject_label, source_path) in enumerate(
        subject_rows, start=1
    ):
        with h5py.File(source_path, "r") as source:
            if decode_attribute(source.attrs["split"]) != "val":
                raise PermissionError(f"Non-validation source encountered: {source_path}")
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
        if reference_db.shape != (2, 793, 463) or mca_db.shape != reference_db.shape:
            raise ValueError(f"Invalid spectral shape for {subject_label}")
        if directions.shape != (793, 6) or frequency.shape != (463,):
            raise ValueError(f"Invalid common grid for {subject_label}")
        if np.count_nonzero(interpolation) != 767 or q26_indices.size != 26:
            raise ValueError(f"Invalid masks for {subject_label}")
        q26_mask = np.zeros(793, dtype=bool)
        q26_mask[q26_indices] = True
        if np.any(q26_mask & interpolation) or not np.array_equal(~q26_mask, interpolation):
            raise ValueError(f"Q26/interpolation partition mismatch for {subject_label}")
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
                    raise ValueError(f"Common grid mismatch: {subject_label} {key}")

        for method, prediction_root in prediction_roots.items():
            prediction_file = prediction_root / "subjects" / subject_label / "prediction.h5"
            with h5py.File(prediction_file, "r") as prediction_handle:
                if decode_attribute(prediction_handle.attrs["split"]) != "val":
                    raise PermissionError(f"Non-validation prediction: {prediction_file}")
                if int(np.asarray(prediction_handle.attrs["subject_id"]).item()) != subject_id:
                    raise ValueError(f"Prediction subject mismatch: {prediction_file}")
                residual_db = np.asarray(
                    prediction_handle["predicted_residual_db"][:], dtype=np.float32
                )
            if residual_db.shape != reference_db.shape or not np.all(np.isfinite(residual_db)):
                raise FloatingPointError(f"Invalid prediction for {method}/{subject_label}")
            predicted_db = mca_db + residual_db
            lsd, direction_lsd, ear_direction_lsd = full_sphere_lsd(
                predicted_db, reference_db, interpolation, directions[:, 5]
            )
            for ear_index, ear_label in enumerate(("left", "right")):
                for direction_index, value in enumerate(ear_direction_lsd[ear_index]):
                    direction_subject_rows.append(
                        {
                            "SubjectLabel": subject_label,
                            "SubjectID": subject_id,
                            "Method": method,
                            "Ear": ear_label,
                            "DirectionIndexZeroBased": direction_index,
                            "IsQ26Input": int(q26_mask[direction_index]),
                            "IncludedInAggregate": int(interpolation[direction_index]),
                            "LSD_dB": format(float(value), ".17g"),
                        }
                    )
            metrics = {"FullSphereLSD": lsd}
            metrics.update(
                spectral_shape_metrics(
                    predicted_db[:, interpolation, :],
                    reference_db[:, interpolation, :],
                    directions[interpolation, 5],
                    frequency,
                )
            )
            band_centers, profile, band_mean = spectral_band_ild_profile(
                predicted_db[:, horizontal, :],
                reference_db[:, horizontal, :],
                directions[horizontal, 5],
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
                        "MethodLabel": method_labels[method],
                        "Endpoint": endpoint,
                        "Unit": units[endpoint],
                        "Value": format(value, ".17g"),
                    }
                )
            band_values[method].append(profile)
            for band_index, (center, value) in enumerate(zip(band_centers, profile)):
                band_subject_rows.append(
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
                direction_lsd, distance, interpolation, directions[:, 5]
            )
            for bin_label, value in bins.items():
                distance_by_subject[bin_label][method][subject_label] = value
                spatial_subject_rows.append(
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
        print(f"secondary metrics [{subject_index}/44] {subject_label}", flush=True)

    assert common_grid is not None
    aggregate_rows: list[dict[str, object]] = []
    for endpoint, methods in scalar_values.items():
        for method, subject_values in methods.items():
            values = [subject_values[label] for label in subject_labels]
            mean, lower, upper = bootstrap_mean_interval(
                values, replicates=bootstrap_replicates, seed=bootstrap_seed
            )
            aggregate_rows.append(
                {
                    "Method": method,
                    "MethodLabel": method_labels[method],
                    "Endpoint": endpoint,
                    "Unit": units[endpoint],
                    "SubjectCount": 44,
                    "Mean": format(mean, ".17g"),
                    "SampleStd": format(float(np.std(values, ddof=1)), ".17g"),
                    "Bootstrap95Lower": format(lower, ".17g"),
                    "Bootstrap95Upper": format(upper, ".17g"),
                }
            )
    for bin_label, methods in distance_by_subject.items():
        endpoint = f"SpatialLSD_{bin_label}deg"
        scalar_values[endpoint] = methods
        units[endpoint] = "dB"
        for method, subject_values in methods.items():
            values = [subject_values[label] for label in subject_labels]
            mean, lower, upper = bootstrap_mean_interval(
                values, replicates=bootstrap_replicates, seed=bootstrap_seed
            )
            spatial_subject_rows.append(
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

    for method in method_specs:
        profiles = np.stack(band_values[method], axis=0)
        for band_index, center in enumerate(band_centers):
            mean, lower, upper = bootstrap_mean_interval(
                profiles[:, band_index], replicates=bootstrap_replicates, seed=bootstrap_seed
            )
            band_subject_rows.append(
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

    tail_rows: list[dict[str, object]] = []
    primary = load_primary_metrics(primary_path, payload["primary_metrics"]["method_mapping"])
    all_tail_endpoints = {**primary, **scalar_values}
    for endpoint, methods in all_tail_endpoints.items():
        if set(methods) != set(method_specs):
            raise ValueError(f"Incomplete methods for tail endpoint {endpoint}")
        unit = "dB" if endpoint in primary else units[endpoint]
        for baseline in ("HYBRID", "FILMENS", "MCAR"):
            statistics = paired_tail_statistics(
                methods["BOUNDED"], methods[baseline],
                tie_tolerance=tie_tolerance,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            tail_rows.append(
                {
                    "Endpoint": endpoint,
                    "Unit": unit,
                    "Baseline": baseline,
                    "Difference": "BOUNDED-minus-baseline; negative favors BOUNDED",
                    **statistics,
                }
            )

    map_rows: list[dict[str, object]] = []
    interpolation_indices = np.flatnonzero(common_grid["interpolation"].astype(bool))
    for method in method_specs:
        mean_map = np.stack(direction_values[method], axis=0).mean(axis=0)
        for local_index, direction_index in enumerate(interpolation_indices):
            features = common_grid["directions"][direction_index]
            map_rows.append(
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

    write_csv(partial_root / "per_subject_metrics.csv", per_subject_rows)
    write_csv(partial_root / "aggregate_metrics.csv", aggregate_rows)
    write_csv(partial_root / "paired_tail_risk.csv", tail_rows)
    write_csv(partial_root / "band_ild_profile.csv", band_subject_rows)
    write_csv(partial_root / "spatial_distance_bins.csv", spatial_subject_rows)
    write_csv(partial_root / "spatial_direction_map.csv", map_rows)
    write_csv(partial_root / "per_subject_direction_lsd.csv", direction_subject_rows)
    quality = {
        "schema_version": "1.0",
        "status": "passed",
        "split": "val",
        "subject_count": 44,
        "test_subject_count_read": 0,
        "method_count": 4,
        "direction_count": 793,
        "q26_direction_count": 26,
        "interpolation_direction_count": 767,
        "horizontal_interpolation_direction_count": 72,
        "frequency_bin_count": 463,
        "band_count_200_to_18000_hz": int(len(band_centers)),
        "all_finite": True,
        "manifest_identity_sha256": identity,
    }
    (partial_root / "quality_checks.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    snapshot = dict(payload)
    snapshot["manifest_path"] = str(manifest_path)
    snapshot["verified_identity_sha256"] = identity
    (partial_root / "protocol_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "val",
        "subject_count": 44,
        "test_subject_count_read": 0,
        "manifest_identity_sha256": identity,
        "methods": list(method_specs),
        "secondary_scalar_endpoints": list(units),
        "primary_endpoints_used_for_tail_analysis": list(primary),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "deferred_unqualified_endpoints": payload["deferred_unqualified_endpoints"],
        "files": sorted(
            [path.name for path in partial_root.iterdir()] + ["summary.json"]
        ),
    }
    (partial_root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    partial_root.replace(output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
