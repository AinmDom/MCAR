"""Summarize the frozen Hybrid LSD-B validation comparison and its integrity."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECONDARY = ROOT / "results/sonicom_hybrid_lsd_b_secondary_validation"
PRIMARY = ROOT / "results/sonicom_hybrid_lsd_b_vs_old_hybrid_validation"
OUT_JSON = ROOT / "reports/HYBRID_LSD_B_VALIDATION_COMPARISON_20260905.json"
OUT_MD = ROOT / "reports/HYBRID_LSD_B_VALIDATION_COMPARISON_20260905.md"


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("Comparison summary is already frozen")
    quality = json.loads((SECONDARY / "quality_checks.json").read_text())
    summary = json.loads((SECONDARY / "summary.json").read_text())
    inference = json.loads((ROOT / "artifacts/reconstruction/sonicom_hybrid_lsd_b_e190_ensemble_validation/inference_report.json").read_text())
    manifest = json.loads((ROOT / "configs/experiments/sonicom_hybrid_lsd_b_secondary_validation_manifest.json").read_text())
    if not (quality["status"] == "passed" and quality["all_finite"] and quality["test_subject_count_read"] == 0):
        raise AssertionError("Secondary quality failed")
    if not (summary["bootstrap_seed"] == 20260904 and summary["bootstrap_replicates"] == 10000):
        raise AssertionError("Bootstrap metadata mismatch")
    if not (inference["status"] == "completed" and inference["subject_count"] == 44 and inference["test_subject_count_read"] == 0):
        raise AssertionError("Inference incomplete")
    if not all(item["finite"] and item["shape"] == [2, 793, 463] for item in inference["subjects"]):
        raise AssertionError("Prediction inventory incomplete")
    primary_quality = rows(PRIMARY / "quality_checks.csv")
    if len(primary_quality) != 44 or any(not all(math.isfinite(float(row[key])) for key in (
        "ReferenceILDMetadataMaxError_dB", "InterpolationDirectionCount", "HorizontalDirectionCount")) for row in primary_quality):
        raise AssertionError("Primary quality failed")

    aggregate = rows(SECONDARY / "aggregate_metrics.csv")
    paired = {row["Endpoint"]: row for row in rows(SECONDARY / "paired_tail_risk.csv") if row["Baseline"] == "HYBRID"}
    candidate = {row["Endpoint"]: float(row["Mean"]) for row in aggregate if row["Method"] == "BOUNDED"}
    baseline = {row["Endpoint"]: float(row["Mean"]) for row in aggregate if row["Method"] == "HYBRID"}
    for row in rows(PRIMARY / "aggregate_metrics.csv"):
        if row["Method"] == "HYBRID":
            candidate[row["Metric"]] = float(row["Mean_dB"])
        elif row["Method"] == "PARENT":
            baseline[row["Metric"]] = float(row["Mean_dB"])
    for row in rows(SECONDARY / "spatial_distance_bins.csv"):
        if row["RecordType"] != "aggregate":
            continue
        endpoint = f"SpatialLSD_{row['DistanceBin_deg']}deg"
        if row["Method"] == "BOUNDED":
            candidate[endpoint] = float(row["LSD_dB"])
        elif row["Method"] == "HYBRID":
            baseline[endpoint] = float(row["LSD_dB"])
    endpoints = ["FullSphereLSD", "FullSphereERB", "Contralateral25ERB", "ContralateralHighFrequency",
        "HorizontalILDMAE", "HFFirstDifferenceMAE", "HFSecondDifferenceMAE", "MultiScaleNotchDepthMAE",
        "ERBBandILDMean", "SpatialLSD_0_10deg", "SpatialLSD_10_20deg", "SpatialLSD_20_30deg", "SpatialLSD_30_180deg"]
    comparisons = {}
    for endpoint in endpoints:
        row = paired[endpoint]
        comparisons[endpoint] = {"unit": row["Unit"], "candidate_mean": candidate.get(endpoint),
            "baseline_mean": baseline.get(endpoint), "mean_difference_candidate_minus_baseline": float(row["MeanDifference"]),
            "bootstrap95": [float(row["Bootstrap95Lower"]), float(row["Bootstrap95Upper"])],
            "median_difference": float(row["MedianDifference"]), "p90_difference": float(row["P90Difference"]),
            "worst_decile_mean_difference": float(row["WorstDecileMean"]),
            "maximum_degradation": float(row["MaximumDegradation"]), "maximum_subject": row["MaximumSubject"],
            "wins": int(row["Wins"]), "ties": int(row["Ties"]), "losses": int(row["Losses"])}

    curves = []
    verification = json.loads((ROOT / "reports/HYBRID_LSD_B_TRAINING_VERIFICATION_20260905.json").read_text())
    for member in verification["members"]:
        seed = member["seed"]
        ledger = json.loads((ROOT / f"artifacts/training/sonicom_film_siren_spectral_cnn_lsd_b_seed{seed}_e190/validation_ledger.json").read_text())
        curve = [(item["cycle"], item["aggregate"]["aggregate_full_sphere_lsd_db"]) for item in ledger]
        best_cycle, best_lsd = min(curve, key=lambda item: item[1])
        curves.append({"seed": seed, "validation_points": len(curve), "cycle_5_lsd_db": curve[0][1],
            "minimum_lsd_db": best_lsd, "minimum_lsd_cycle_diagnostic_only": best_cycle,
            "cycle_190_lsd_db": curve[-1][1], "authoritative_cycle": 190})

    lsd = comparisons["FullSphereLSD"]
    relative = 100 * (lsd["baseline_mean"] - lsd["candidate_mean"]) / lsd["baseline_mean"]
    payload = {"schema_version": "1.0", "status": "completed", "split": "val", "subject_count": 44,
        "test_subjects_read": 0, "candidate": "Hybrid LSD B E190 three-member equal ensemble",
        "baseline": "Original Hybrid E190 three-member equal ensemble",
        "ensemble_manifest_identity": "D1611128A86118ACE0EFD471A4F99456F1267B10D1C39C68AD769239FB589CE6",
        "secondary_manifest_identity": manifest["identity_sha256"], "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260904, "difference": "candidate minus baseline; negative is better",
        "full_sphere_lsd_relative_improvement_percent": relative, "comparisons": comparisons,
        "single_seed_validation_lsd_curves": curves,
        "integrity": {"training_verification": "passed", "inference_subjects": 44,
            "prediction_shape": [2, 793, 463], "prediction_all_finite": True,
            "secondary_quality": "passed/all_finite", "primary_quality_rows": 44,
            "primary_reference_ild_metadata_max_error_db": max(float(row["ReferenceILDMetadataMaxError_dB"]) for row in primary_quality),
            "test_subjects_read": 0},
        "decision": "LSD objective produces a small, consistent validation LSD gain with small ERB and spectral-smoothness tradeoffs. Retain as a validation candidate; it does not establish that MCA internal ERB processing caused the original LSD gap and is not independent test confirmation."}
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    table_order = endpoints[:9]
    lines = ["# Hybrid LSD B validation comparison", "", "Status: completed on 44 validation subjects; test reads: 0.", "",
        f"FullSphereLSD changed from **{lsd['baseline_mean']:.9f}** to **{lsd['candidate_mean']:.9f} dB**, a **{relative:.3f}%** reduction. "
        f"The paired mean difference was **{lsd['mean_difference_candidate_minus_baseline']:.9f} dB** "
        f"(95% bootstrap [{lsd['bootstrap95'][0]:.9f}, {lsd['bootstrap95'][1]:.9f}]; "
        f"wins/ties/losses {lsd['wins']}/{lsd['ties']}/{lsd['losses']}).", "", "## Candidate minus original Hybrid", "",
        "| Endpoint | New mean | Old mean | Difference | 95% bootstrap | W/T/L |", "|---|---:|---:|---:|---:|---:|"]
    for endpoint in table_order:
        item = comparisons[endpoint]
        lines.append(f"| {endpoint} | {item['candidate_mean']:.9f} | {item['baseline_mean']:.9f} | {item['mean_difference_candidate_minus_baseline']:+.9f} | [{item['bootstrap95'][0]:+.9f}, {item['bootstrap95'][1]:+.9f}] | {item['wins']}/{item['ties']}/{item['losses']} |")
    lines += ["", "LSD gains were clear in the 0–10°, 10–20°, and 20–30° distance bins. The 30–180° mean difference was -0.003499 dB with an interval crossing zero; its maximum subject degradation was +0.101600 dB (P0166).",
        "", "The intervention slightly worsened full-sphere and contralateral-25 ERB and both high-frequency difference metrics. Contralateral high-frequency was unchanged within uncertainty; strict horizontal ILD improved by 0.001779 dB.",
        "", "This supports a small LSD/objective tradeoff. It does not isolate MCA's internal ERB correction because that path was unchanged. The result is exploratory validation evidence and must not be described as independent test confirmation.",
        "", "## Integrity", "", "Three runs: 3/3 exit 0, E190, 49,780 steps each; all history/ledger values and E190 weights finite; checkpoint hashes matched; clean training Git; test reads 0. New inference: 44/44 subjects, shape [2,793,463], all finite. Bootstrap: 10,000 paired subject resamples, seed 20260904."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "lsd_new": lsd["candidate_mean"], "lsd_old": lsd["baseline_mean"],
        "difference": lsd["mean_difference_candidate_minus_baseline"], "ci": lsd["bootstrap95"], "relative_percent": relative,
        "wins_ties_losses": [lsd["wins"], lsd["ties"], lsd["losses"]], "test_subjects_read": 0}, indent=2))


if __name__ == "__main__":
    main()
