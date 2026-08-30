"""Evaluate preregistered gate, ITD, and dominant-notch endpoints on validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

from mcar.evaluation.deferred_secondary_metrics import (
    absolute_distribution_summary,
    dominant_notch_location_metrics,
    enable_external_torchaudio,
    strict_hrir_from_selected_db,
)
from mcar.evaluation.secondary_metrics import (
    DISTANCE_BIN_EDGES_DEG,
    DISTANCE_BIN_LABELS,
    bootstrap_mean_interval,
    minimum_q26_distance_degrees,
    normalized_direction_weights,
    paired_tail_statistics,
)
from mcar.fsp_ae_signal import estimate_itd_seconds
from mcar.paths import project_root
from mcar.predictors import BoundedMcarFilmCorrectionPredictor
from mcar.training.train_film_siren import file_sha256, split_subject_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def manifest_identity(payload: dict[str, Any]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256")).upper()
    actual = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Manifest identity mismatch")
    return actual


def inventory_digest(root: Path, subjects: list[str], filename: str) -> str:
    digest = hashlib.sha256()
    for subject in sorted(subjects):
        path = root / "subjects" / subject / filename
        digest.update(subject.encode("ascii") + b"\0")
        digest.update(file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def strict_metadata(handle: h5py.File) -> dict[str, np.ndarray | int]:
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
        "single_sided_frequency_count": int(
            np.asarray(group.attrs["single_sided_frequency_count"]).item()
        ),
        "hrir_length": int(np.asarray(group.attrs["hrir_length"]).item()),
    }


def itd_values(hrir_direction_ear_time: torch.Tensor, settings: dict[str, Any]) -> np.ndarray:
    estimates = estimate_itd_seconds(
        hrir_direction_ear_time,
        float(settings["sampling_rate_hz"]),
        upsampled_rate_hz=float(settings["upsampled_rate_hz"]),
        lowpass_hz=float(settings["lowpass_hz"]),
        maximum_itd_seconds=float(settings["maximum_itd_seconds"]),
        direction_batch_size=int(settings["direction_batch_size"]),
    )
    return estimates.cpu().numpy().astype(np.float64)


def masks_by_region(
    interpolation: np.ndarray, distance: np.ndarray
) -> list[tuple[str, np.ndarray]]:
    output = [("all", interpolation.copy())]
    for index, label in enumerate(DISTANCE_BIN_LABELS):
        lower, upper = DISTANCE_BIN_EDGES_DEG[index : index + 2]
        mask = interpolation & (distance >= lower)
        mask &= distance <= upper + 1e-10 if index == 3 else distance < upper
        output.append((f"distance_{label}", mask))
    return output


def append_mechanism_rows(
    rows: list[dict[str, object]],
    subject: str,
    entity: str,
    quantity: str,
    values: np.ndarray,
    interpolation: np.ndarray,
    distance: np.ndarray,
    *,
    is_gate: bool,
) -> None:
    for region, direction_mask in masks_by_region(interpolation, distance):
        for ear_name, ear_slice in (("both", slice(None)), ("left", 0), ("right", 1)):
            subset = values[ear_slice, direction_mask, :]
            summary = absolute_distribution_summary(subset, gate=is_gate)
            for statistic, value in summary.items():
                rows.append(
                    {
                        "SubjectLabel": subject,
                        "Entity": entity,
                        "Quantity": quantity,
                        "Region": region,
                        "Ear": ear_name,
                        "Statistic": statistic,
                        "Value": format(value, ".17g"),
                    }
                )


def main() -> None:
    args = parse_arguments()
    root = project_root()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest_identity(payload)
    if payload["status"] != "frozen" or payload["dataset"]["split"] != "val":
        raise PermissionError("Deferred metrics are validation-only")
    for resource in payload["implementation"]["resources"]:
        if file_sha256(root / resource["path"]) != resource["sha256"]:
            raise ValueError(f"Implementation hash mismatch: {resource['path']}")
    for resource in payload["implementation"]["external_resources"]:
        if file_sha256(Path(resource["path"])) != resource["sha256"]:
            raise ValueError(f"External dependency hash mismatch: {resource['path']}")
    torchaudio_version = enable_external_torchaudio(
        Path(payload["implementation"]["torchaudio_site_packages"])
    )
    if torchaudio_version != payload["implementation"]["torchaudio_version"]:
        raise RuntimeError("torchaudio version mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen diagnostics and ITD")

    dataset_root = (root / payload["dataset"]["root"]).resolve()
    split_csv = (root / payload["dataset"]["split_csv"]).resolve()
    subject_rows = split_subject_paths(dataset_root, split_csv, "val")
    subject_labels = [row[1] for row in subject_rows]
    if len(subject_rows) != 44 or len(set(subject_labels)) != 44:
        raise ValueError("Expected 44 unique validation subjects")
    method_specs = payload["methods"]
    prediction_roots: dict[str, Path] = {}
    for method, spec in method_specs.items():
        prediction_root = (root / spec["prediction_root"]).resolve()
        if inventory_digest(prediction_root, subject_labels, "prediction.h5") != spec[
            "prediction_inventory_sha256"
        ]:
            raise ValueError(f"Prediction inventory mismatch: {method}")
        prediction_roots[method] = prediction_root

    model_manifest = json.loads((root / payload["bounded_manifest"]["path"]).read_text())
    if model_manifest["identity_sha256"] != payload["bounded_manifest"]["identity_sha256"]:
        raise ValueError("Bounded ensemble identity mismatch")
    predictors = []
    for member in model_manifest["members"]:
        checkpoint = root / member["checkpoint"]
        if file_sha256(checkpoint) != member["checkpoint_sha256"]:
            raise ValueError("Bounded checkpoint hash mismatch")
        predictors.append(
            BoundedMcarFilmCorrectionPredictor(
                checkpoint,
                split_csv,
                root / model_manifest["dataset"]["q26_csv"],
                root / model_manifest["dataset"]["q26_normalization"],
                device=torch.device("cuda"),
                directions_per_block=int(payload["gate"]["directions_per_block"]),
                allow_test=False,
            )
        )
    weights = np.asarray(model_manifest["ensemble"]["weights"], dtype=np.float64)
    results_root = root / payload["outputs"]["results_root"]
    artifact_root = root / payload["outputs"]["diagnostics_root"]
    result_partial = results_root.with_name(results_root.name + ".partial")
    artifact_partial = artifact_root.with_name(artifact_root.name + ".partial")
    if any(path.exists() for path in (results_root, artifact_root, result_partial, artifact_partial)):
        raise FileExistsError("Refusing to overwrite deferred metric outputs")
    result_partial.mkdir(parents=True)
    artifact_partial.mkdir(parents=True)

    endpoint_rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    spearman_rows: list[dict[str, object]] = []
    endpoint_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    max_prediction_difference = 0.0
    itd_settings = payload["itd"]
    for subject_index, (subject_id, subject_label, source_path) in enumerate(subject_rows, 1):
        with h5py.File(source_path, "r") as source:
            split_value = np.asarray(source.attrs["split"]).item()
            if isinstance(split_value, bytes):
                split_value = split_value.decode()
            if str(split_value) != "val":
                raise PermissionError("Non-validation source encountered")
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
        q26_mask = np.zeros(793, dtype=bool)
        q26_mask[q26_indices] = True
        distance = minimum_q26_distance_degrees(directions[:, 2:5], q26_mask)
        direction_weights = directions[:, 5]
        normalized_interpolation_weights = normalized_direction_weights(
            direction_weights[interpolation]
        )
        sofa_path = root / payload["dataset"]["reference_sofa_root"] / (
            f"{subject_label}_FreeFieldCompMinPhase_44kHz.sofa"
        )
        with h5py.File(sofa_path, "r") as sofa:
            reference_hrir = np.asarray(sofa["Data.IR"][:], dtype=np.float32)
            source_position = np.asarray(sofa["SourcePosition"][:], dtype=np.float64)
        if reference_hrir.shape != (793, 2, 256):
            raise ValueError("Reference SOFA HRIR shape mismatch")
        position_error = float(np.max(np.abs(source_position[:, :2] - directions[:, :2])))
        if position_error > float(payload["dataset"]["coordinate_tolerance_degrees"]):
            raise ValueError("SOFA/dataset direction mapping mismatch")
        reference_itd = itd_values(torch.from_numpy(reference_hrir).cuda(), itd_settings)

        method_residuals: dict[str, np.ndarray] = {}
        for method, prediction_root in prediction_roots.items():
            prediction_file = prediction_root / "subjects" / subject_label / "prediction.h5"
            with h5py.File(prediction_file, "r") as prediction:
                residual = np.asarray(prediction["predicted_residual_db"][:], dtype=np.float32)
            method_residuals[method] = residual
            predicted_db = mca_db + residual
            notch = dominant_notch_location_metrics(
                predicted_db[:, interpolation, :],
                reference_db[:, interpolation, :],
                direction_weights[interpolation],
                frequency,
                matching_tolerance_hz=float(payload["notch"]["matching_tolerance_hz"]),
            )
            hrir = strict_hrir_from_selected_db(predicted_db, metadata)
            candidate_itd = itd_values(hrir.permute(1, 0, 2).cuda(), itd_settings)
            itd_error_us = np.abs(candidate_itd - reference_itd) * 1e6
            metrics = {
                **notch,
                "ITDWeightedMAE_us": float(
                    np.sum(itd_error_us[interpolation] * normalized_interpolation_weights)
                ),
                "ITDMaximumAbsoluteError_us": float(np.max(itd_error_us[interpolation])),
            }
            for endpoint, value in metrics.items():
                if not np.isfinite(value):
                    raise FloatingPointError(f"Non-finite {endpoint}: {method}/{subject_label}")
                endpoint_values[endpoint][method][subject_label] = value
                endpoint_rows.append(
                    {
                        "SubjectLabel": subject_label,
                        "SubjectID": subject_id,
                        "Method": method,
                        "Endpoint": endpoint,
                        "Value": format(value, ".17g"),
                    }
                )

        prepared = predictors[0].prepare_subject(source_path)
        diagnostics = [p.predict_prepared_diagnostics(prepared) for p in predictors]
        member_final = np.stack([item.final_residual_db for item in diagnostics])
        member_base = np.stack([item.base_residual_db for item in diagnostics])
        member_correction = np.stack([item.applied_correction_db for item in diagnostics])
        member_gate = np.stack([item.gate for item in diagnostics])
        ensemble_final = np.average(member_final, axis=0, weights=weights)
        ensemble_base = np.average(member_base, axis=0, weights=weights)
        ensemble_correction = np.average(member_correction, axis=0, weights=weights)
        difference = float(np.max(np.abs(ensemble_final - method_residuals["BOUNDED"])))
        max_prediction_difference = max(max_prediction_difference, difference)
        if difference > float(payload["gate"]["prediction_tolerance_db"]):
            raise AssertionError("Regenerated bounded prediction mismatch")
        subject_artifact = artifact_partial / "subjects" / subject_label / "diagnostics.h5"
        subject_artifact.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(subject_artifact, "w") as handle:
            handle.create_dataset("member_gate", data=member_gate, compression="gzip")
            handle.create_dataset(
                "member_applied_correction_db", data=member_correction, compression="gzip"
            )
            handle.create_dataset(
                "ensemble_applied_correction_db", data=ensemble_correction, compression="gzip"
            )
            handle.attrs["split"] = "val"
            handle.attrs["subject_id"] = subject_id
            handle.attrs["manifest_identity_sha256"] = identity
        for member_index in range(3):
            append_mechanism_rows(
                mechanism_rows, subject_label, f"member_{member_index + 1}", "gate",
                member_gate[member_index], interpolation, distance, is_gate=True
            )
            append_mechanism_rows(
                mechanism_rows, subject_label, f"member_{member_index + 1}", "correction_db",
                member_correction[member_index], interpolation, distance, is_gate=False
            )
        append_mechanism_rows(
            mechanism_rows, subject_label, "deployed_ensemble", "correction_db",
            ensemble_correction, interpolation, distance, is_gate=False
        )
        benefit = np.abs(reference_db - (mca_db + ensemble_base)) - np.abs(
            reference_db - (mca_db + ensemble_final)
        )
        rho = float(
            spearmanr(
                np.abs(ensemble_correction[:, interpolation, :]).reshape(-1),
                benefit[:, interpolation, :].reshape(-1),
            ).statistic
        )
        if not np.isfinite(rho):
            raise FloatingPointError("Non-finite subject Spearman correlation")
        spearman_rows.append({"SubjectLabel": subject_label, "SpearmanRho": format(rho, ".17g")})
        print(f"deferred secondary [{subject_index}/44] {subject_label}", flush=True)

    aggregate_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for endpoint, methods in endpoint_values.items():
        for method, by_subject in methods.items():
            values = [by_subject[label] for label in subject_labels]
            mean, lower, upper = bootstrap_mean_interval(values)
            aggregate_rows.append(
                {"Method": method, "Endpoint": endpoint, "SubjectCount": 44,
                 "Mean": format(mean, ".17g"), "SampleStd": format(np.std(values, ddof=1), ".17g"),
                 "Bootstrap95Lower": format(lower, ".17g"),
                 "Bootstrap95Upper": format(upper, ".17g")}
            )
        if endpoint.startswith("ITD") or endpoint.startswith("DominantNotch"):
            paired_rows.append(
                {"Endpoint": endpoint, "Baseline": "MCAR", **paired_tail_statistics(
                    methods["BOUNDED"], methods["MCAR"]
                )}
            )
    mechanism_groups: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in mechanism_rows:
        key = tuple(str(row[name]) for name in ("Entity", "Quantity", "Region", "Ear", "Statistic"))
        mechanism_groups[key].append(float(row["Value"]))
    mechanism_aggregate: list[dict[str, object]] = []
    for key, values in mechanism_groups.items():
        mean, lower, upper = bootstrap_mean_interval(values)
        mechanism_aggregate.append(
            {"Entity": key[0], "Quantity": key[1], "Region": key[2], "Ear": key[3],
             "Statistic": key[4], "SubjectCount": 44, "Mean": format(mean, ".17g"),
             "Bootstrap95Lower": format(lower, ".17g"),
             "Bootstrap95Upper": format(upper, ".17g")}
        )
    rho_values = np.asarray([float(row["SpearmanRho"]) for row in spearman_rows])
    spearman_summary = {
        "subject_count": 44,
        "median": float(np.median(rho_values)),
        "q25": float(np.quantile(rho_values, 0.25, method="linear")),
        "q75": float(np.quantile(rho_values, 0.75, method="linear")),
    }
    write_csv(result_partial / "per_subject_endpoints.csv", endpoint_rows)
    write_csv(result_partial / "aggregate_endpoints.csv", aggregate_rows)
    write_csv(result_partial / "paired_vs_mcar.csv", paired_rows)
    write_csv(result_partial / "mechanism_per_subject.csv", mechanism_rows)
    write_csv(result_partial / "mechanism_aggregate.csv", mechanism_aggregate)
    write_csv(result_partial / "correction_benefit_spearman.csv", spearman_rows)
    (result_partial / "correction_benefit_spearman_summary.json").write_text(
        json.dumps(spearman_summary, indent=2, allow_nan=False) + "\n"
    )
    quality = {
        "status": "passed", "split": "val", "subject_count": 44,
        "method_count": 4, "test_subject_count_read": 0, "all_finite": True,
        "direction_count": 793, "interpolation_direction_count": 767,
        "frequency_bin_count": 463,
        "bounded_prediction_max_abs_difference_db": max_prediction_difference,
        "manifest_identity_sha256": identity,
        "localization_status": "not_run_dependency_unavailable",
    }
    (result_partial / "quality_checks.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False) + "\n"
    )
    (result_partial / "protocol_snapshot.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "status": "completed", "split": "val", "subject_count": 44,
        "test_subject_count_read": 0, "endpoints": sorted(endpoint_values),
        "manifest_identity_sha256": identity,
        "localization_status": "blocked_missing_frozen_official_model",
    }
    (result_partial / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    result_partial.replace(results_root)
    artifact_partial.replace(artifact_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
