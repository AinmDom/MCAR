"""Assemble the frozen paired FSC spectral-CNN ablation on validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from mcar.evaluation.deferred_secondary_metrics import (
    enable_external_torchaudio,
    strict_hrir_from_selected_db,
)
from mcar.evaluation.secondary_metrics import (
    full_sphere_lsd,
    normalized_direction_weights,
    paired_tail_statistics,
    spectral_band_ild_profile,
    spectral_shape_metrics,
)
from mcar.fsp_ae_signal import estimate_itd_seconds
from mcar.paths import project_root
from mcar.training.train_film_siren import split_subject_paths
from mcar.training.train_film_siren_stage_c import horizontal_interpolation_indices


METRICS = (
    "FullSphereERB",
    "Contralateral25ERB",
    "ERBBandILDMean",
    "ITDWeightedMAE",
    "FullSphereLSD",
    "HFFirstDifferenceMAE",
    "HFSecondDifferenceMAE",
    "MultiScaleNotchDepthMAE",
)
UNITS = {
    "FullSphereERB": "dB",
    "Contralateral25ERB": "dB",
    "ERBBandILDMean": "dB",
    "ITDWeightedMAE": "us",
    "FullSphereLSD": "dB",
    "HFFirstDifferenceMAE": "dB/bin",
    "HFSecondDifferenceMAE": "dB/bin^2",
    "MultiScaleNotchDepthMAE": "dB",
}
VARIANTS = ("FiLM-SIREN E130", "FSC E190")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def decode(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strict_metadata(handle: h5py.File) -> dict[str, object]:
    group = handle["strict_ild"]
    return {
        "mca_selected_phase_rad": np.asarray(group["mca_selected_phase_rad"][:]),
        "mca_outside_real": np.asarray(group["mca_outside_real"][:]),
        "mca_outside_imag": np.asarray(group["mca_outside_imag"][:]),
        "selected_bin_indices_zero_based": np.asarray(group["selected_bin_indices_zero_based"][:]),
        "outside_bin_indices_zero_based": np.asarray(group["outside_bin_indices_zero_based"][:]),
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


def verify_manifest(root: Path, payload: dict[str, Any]) -> None:
    if payload["status"] != "frozen" or payload["dataset"]["split"] != "val":
        raise PermissionError("The ablation is frozen to validation only")
    if payload["dataset"]["test_access_allowed"] or payload["dataset"]["subject_count"] != 44:
        raise PermissionError("Invalid validation/test boundary")
    for resource in payload["resources"]:
        if sha256(root / resource["path"]) != resource["sha256"]:
            raise ValueError(f"Resource hash mismatch: {resource['path']}")
    pairs = payload["checkpoint_pairs"]
    if [pair["seed"] for pair in pairs] != [20260821, 20260822, 20260823]:
        raise ValueError("Formal seed set changed")
    for pair in pairs:
        for key in ("e130", "e190"):
            spec = pair[key]
            if Path(spec["path"]).name != "last.pt" or sha256(root / spec["path"]) != spec["sha256"]:
                raise ValueError(f"Checkpoint identity mismatch: {pair['seed']} {key}")
            report = json.loads((root / spec["training_report"]).read_text(encoding="utf-8"))
            expected_cycle = 130 if key == "e130" else 190
            if report["status"] != "completed" or report["cycles"] != expected_cycle:
                raise ValueError(f"Training completion mismatch: {pair['seed']} {key}")
            if report["test_subjects_read"] != 0 or report["authoritative_checkpoint"] != "last.pt":
                raise PermissionError(f"Checkpoint policy mismatch: {pair['seed']} {key}")
        report190 = json.loads((root / pair["e190"]["training_report"]).read_text(encoding="utf-8"))
        if report190["initial_film_checkpoint"]["sha256"] != pair["e130"]["sha256"]:
            raise ValueError(f"E190 is not paired to same-seed E130: {pair['seed']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    root = project_root()
    payload = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    verify_manifest(root, payload)
    enable_external_torchaudio(Path(payload["runtime"]["torchaudio_site_packages"]))

    dataset_root = root / payload["dataset"]["root"]
    split_csv = root / payload["dataset"]["split_csv"]
    subjects = split_subject_paths(dataset_root, split_csv, "val")
    labels = [item[1] for item in subjects]
    if len(labels) != 44 or len(set(labels)) != 44:
        raise ValueError("Expected 44 unique validation subjects")

    predictions = {(int(x["seed"]), x["variant"]): x for x in payload["predictions"]}
    if set(predictions) != {(s, v) for s in (20260821, 20260822, 20260823) for v in VARIANTS}:
        raise ValueError("Prediction pairing matrix is incomplete")
    for key, spec in predictions.items():
        report_path = root / spec["root"] / "inference_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "completed" or report["split"] != "val":
            raise ValueError(f"Prediction report incomplete: {key}")
        if report["subject_count"] != 44 or report["test_subject_count_read"] != 0:
            raise PermissionError(f"Prediction split violation: {key}")
        expected = next(pair for pair in payload["checkpoint_pairs"] if pair["seed"] == key[0])
        ckpt = expected["e130" if key[1] == VARIANTS[0] else "e190"]
        report_checkpoint = report.get("checkpoint_sha256")
        if report_checkpoint is None:
            report_checkpoint = report.get("single_member", {}).get("checkpoint_sha256")
        if report_checkpoint != ckpt["sha256"]:
            raise ValueError(f"Prediction checkpoint mismatch: {key}")
        if {row["subject_label"] for row in report["subjects"]} != set(labels):
            raise ValueError(f"Prediction subject inventory mismatch: {key}")

    primary_path = root / payload["output_root"] / "primary_subject_level.csv"
    primary: dict[tuple[int, str, str, str], float] = {}
    with primary_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            primary[(int(row["Seed"]), row["SubjectLabel"], row["Variant"], row["Metric"])] = float(row["Value"])
    if len(primary) != 3 * 44 * 2 * 2 or not np.all(np.isfinite(list(primary.values()))):
        raise ValueError("Primary metric table is incomplete or non-finite")

    values: dict[tuple[int, str, str], dict[str, float]] = defaultdict(dict)
    rows: list[dict[str, object]] = []
    itd_settings = payload["itd"]
    sofa_root = root / payload["dataset"]["reference_sofa_root"]
    for subject_index, (subject_id, label, source_path) in enumerate(subjects, 1):
        with h5py.File(source_path, "r") as source:
            if decode(source.attrs["split"]) != "val":
                raise PermissionError("Non-validation source encountered")
            reference_db = np.asarray(source["reference_logmag_db"][:], dtype=np.float32)
            mca_db = np.asarray(source["mca_logmag_db"][:], dtype=np.float32)
            directions = np.asarray(source["direction_features"][:], dtype=np.float32)
            frequency = np.asarray(source["frequency_hz"][:], dtype=np.float32).reshape(-1)
            interpolation = np.asarray(source["interpolation_evaluation_mask"][:], dtype=bool).reshape(-1)
            metadata = strict_metadata(source)
        if reference_db.shape != (2, 793, 463) or np.count_nonzero(interpolation) != 767:
            raise ValueError(f"Formal Q26 source mismatch: {label}")
        horizontal = horizontal_interpolation_indices(directions, interpolation)
        if horizontal.size != 72:
            raise ValueError(f"Horizontal grid mismatch: {label}")
        with h5py.File(sofa_root / f"{label}_FreeFieldCompMinPhase_44kHz.sofa", "r") as sofa:
            reference_hrir = np.asarray(sofa["Data.IR"][:], dtype=np.float32)
        reference_itd = itd_values(reference_hrir, itd_settings)
        weights = normalized_direction_weights(directions[interpolation, 5])

        for seed in (20260821, 20260822, 20260823):
            for variant in VARIANTS:
                prediction_path = root / predictions[(seed, variant)]["root"] / "subjects" / label / "prediction.h5"
                with h5py.File(prediction_path, "r") as prediction:
                    if decode(prediction.attrs["split"]) != "val":
                        raise PermissionError("Non-validation prediction encountered")
                    residual = np.asarray(prediction["predicted_residual_db"][:], dtype=np.float32)
                if residual.shape != reference_db.shape or not np.all(np.isfinite(residual)):
                    raise FloatingPointError(f"Invalid prediction: {seed}/{variant}/{label}")
                predicted_db = mca_db + residual
                lsd, _, _ = full_sphere_lsd(predicted_db, reference_db, interpolation, directions[:, 5])
                computed = {"FullSphereLSD": lsd}
                computed.update(spectral_shape_metrics(
                    predicted_db[:, interpolation, :], reference_db[:, interpolation, :],
                    directions[interpolation, 5], frequency,
                ))
                _, _, computed["ERBBandILDMean"] = spectral_band_ild_profile(
                    predicted_db[:, horizontal, :], reference_db[:, horizontal, :],
                    directions[horizontal, 5], frequency,
                )
                predicted_hrir = strict_hrir_from_selected_db(predicted_db, metadata)
                predicted_itd = itd_values(
                    predicted_hrir.permute(1, 0, 2).cpu().numpy(), itd_settings
                )
                computed["ITDWeightedMAE"] = float(
                    np.sum(np.abs(predicted_itd[interpolation] - reference_itd[interpolation]) * 1e6 * weights)
                )
                for metric in METRICS:
                    value = primary[(seed, label, variant, metric)] if metric in METRICS[:2] else computed[metric]
                    values[(seed, variant, metric)][label] = value
                    rows.append({
                        "Seed": seed, "SubjectLabel": label, "SubjectID": subject_id,
                        "Variant": variant, "Metric": metric, "Unit": UNITS[metric],
                        "Value": format(value, ".17g"),
                    })
        print(f"FSC CNN ablation secondary/ITD [{subject_index}/44] {label}", flush=True)

    if len(rows) != 2112 or not np.all(np.isfinite([float(row["Value"]) for row in rows])):
        raise AssertionError("Expected 2112 finite subject-level rows")
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    paper_rows: list[dict[str, object]] = []
    bootstrap = payload["bootstrap"]
    for seed in (20260821, 20260822, 20260823):
        for metric in METRICS:
            variant_summary: dict[str, tuple[float, float]] = {}
            for variant in VARIANTS:
                data = np.asarray([values[(seed, variant, metric)][label] for label in labels])
                variant_summary[variant] = (float(data.mean()), float(data.std(ddof=1)))
                summary_rows.append({
                    "Seed": seed, "Variant": variant, "Metric": metric, "Unit": UNITS[metric],
                    "SubjectCount": 44, "Mean": format(data.mean(), ".17g"),
                    "SampleSD": format(data.std(ddof=1), ".17g"),
                })
            stats = paired_tail_statistics(
                values[(seed, VARIANTS[1], metric)], values[(seed, VARIANTS[0], metric)],
                replicates=int(bootstrap["replicates"]), seed=int(bootstrap["seed"]),
                tie_tolerance=float(bootstrap["tie_tolerance"]),
            )
            paired = {
                "Seed": seed, "Metric": metric, "Unit": UNITS[metric],
                "Difference": "FSC E190 - FiLM-SIREN E130", **stats,
                "BootstrapReplicates": int(bootstrap["replicates"]),
                "BootstrapSeed": int(bootstrap["seed"]),
            }
            paired_rows.append(paired)
            paper_rows.append({
                "Seed": seed, "Metric": metric, "Unit": UNITS[metric],
                "FiLM-SIREN_E130_MeanPlusMinusSampleSD": f"{variant_summary[VARIANTS[0]][0]:.6f} ± {variant_summary[VARIANTS[0]][1]:.6f}",
                "FSC_E190_MeanPlusMinusSampleSD": f"{variant_summary[VARIANTS[1]][0]:.6f} ± {variant_summary[VARIANTS[1]][1]:.6f}",
                "PairedDeltaMean": format(stats["MeanDifference"], ".17g"),
                "PairedBootstrap95Lower": format(stats["Bootstrap95Lower"], ".17g"),
                "PairedBootstrap95Upper": format(stats["Bootstrap95Upper"], ".17g"),
            })

    output = root / payload["output_root"]
    for name in ("subject_level.csv", "seed_summary_mean_std.csv", "paired_bootstrap.csv", "paper_ablation_table.csv", "summary.json", "README.md"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output / name}")
    write_csv(output / "subject_level.csv", rows)
    write_csv(output / "seed_summary_mean_std.csv", summary_rows)
    write_csv(output / "paired_bootstrap.csv", paired_rows)
    write_csv(output / "paper_ablation_table.csv", paper_rows)
    local_metrics = METRICS[-3:]
    local_consistent = all(
        next(row for row in paired_rows if row["Seed"] == seed and row["Metric"] == metric)["MeanDifference"] < 0
        for seed in (20260821, 20260822, 20260823) for metric in local_metrics
    )
    summary = {
        "schema_version": "1.0", "status": "completed", "split": "val",
        "subject_count": 44, "seed_count": 3, "variant_count": 2, "metric_count": 8,
        "subject_level_row_count": len(rows), "expected_subject_level_row_count": 2112,
        "all_finite": True, "test_subjects_read": 0, "strict_pairing": True,
        "checkpoint_policy": "pre-frozen E130/E190 last.pt; no reselection",
        "training_performed": False, "historical_results_overwritten": False,
        "bootstrap": bootstrap, "local_spectral_metrics_all_seed_means_improved": local_consistent,
        "manifest": str(args.manifest.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Formal FSC spectral-CNN ablation (Q26)\n\n"
        "Validation-only paired comparison of fixed `last.pt` checkpoints: FiLM-SIREN E130 versus "
        "the same-seed FSC E190 spectral-CNN refinement. All eight endpoints are lower-is-better; "
        "paired delta is `FSC E190 - FiLM-SIREN E130`, so negative values favor the CNN stage.\n\n"
        "The three seeds are analyzed separately over the same 44 validation subjects. Confidence "
        "intervals use the repository `paired_tail_statistics` implementation with 10,000 bootstrap "
        "replicates and the manifest-fixed seed. No test subject was read.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
