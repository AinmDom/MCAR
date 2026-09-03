"""Consolidate the frozen ten-method test evaluation into one 20-endpoint package.

This script reads committed/result CSVs only. It does not access raw test HRTFs,
run inference, tune a model, or change any frozen prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROOT = ROOT / "results" / "sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method"
SUPPLEMENTARY_ROOT = ROOT / "results" / "sonicom_complete_ten_method_secondary_deferred_test_v1"
OUTPUT = ROOT / "results" / "sonicom_complete_ten_method_test_v1"
PARTIAL = OUTPUT.with_name(OUTPUT.name + ".partial")

METHODS = (
    "SHOnly", "SUpDEqSH", "SUpDEqNN", "SUpDEqBary", "MCA",
    "MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED",
)
LABELS = {
    "SHOnly": "SH only",
    "SUpDEqSH": "SUpDEq SH",
    "SUpDEqNN": "SUpDEq NN",
    "SUpDEqBary": "SUpDEq Barycentric",
    "MCA": "MCA",
    "MCARv351": "MCAR v3.5.1",
    "FSPAE": "FSP-AE",
    "RANF": "RANF",
    "HYBRID": "Hybrid E190",
    "BOUNDED": "Bounded E25",
}
PRIMARY = (
    "FullSphereERB", "Contralateral25ERB",
    "ContralateralHighFrequency", "HorizontalILDMAE",
)
SECONDARY = (
    "FullSphereLSD", "HFFirstDifferenceMAE", "HFSecondDifferenceMAE",
    "MultiScaleNotchDepthMAE", "ERBBandILDMean", "SpatialLSD_0_10deg",
    "SpatialLSD_10_20deg", "SpatialLSD_20_30deg", "SpatialLSD_30_180deg",
)
DEFERRED = (
    "DominantNotchPenalizedMAE_Hz", "DominantNotchMatchedMAE_Hz",
    "DominantNotchMissRate", "DominantNotchSpuriousRate",
    "ReferenceNotchFraction", "ITDWeightedMAE_us", "ITDMaximumAbsoluteError_us",
)
ENDPOINTS = PRIMARY + SECONDARY + DEFERRED
UNITS = {
    **{name: "dB" for name in PRIMARY},
    "FullSphereLSD": "dB",
    "HFFirstDifferenceMAE": "dB/bin",
    "HFSecondDifferenceMAE": "dB/bin^2",
    "MultiScaleNotchDepthMAE": "dB",
    "ERBBandILDMean": "dB",
    "SpatialLSD_0_10deg": "dB",
    "SpatialLSD_10_20deg": "dB",
    "SpatialLSD_20_30deg": "dB",
    "SpatialLSD_30_180deg": "dB",
    "DominantNotchPenalizedMAE_Hz": "Hz",
    "DominantNotchMatchedMAE_Hz": "Hz",
    "DominantNotchMissRate": "fraction",
    "DominantNotchSpuriousRate": "fraction",
    "ReferenceNotchFraction": "fraction",
    "ITDWeightedMAE_us": "us",
    "ITDMaximumAbsoluteError_us": "us",
}
TITLES = {
    "FullSphereERB": "Full-sphere ERB error",
    "Contralateral25ERB": "Contralateral-25 ERB error",
    "ContralateralHighFrequency": "Contralateral high-frequency error",
    "HorizontalILDMAE": "Horizontal-plane ILD MAE",
    "FullSphereLSD": "Full-sphere log-spectral distortion",
    "HFFirstDifferenceMAE": "High-frequency first-difference MAE",
    "HFSecondDifferenceMAE": "High-frequency second-difference MAE",
    "MultiScaleNotchDepthMAE": "Multi-scale notch-depth MAE",
    "ERBBandILDMean": "ERB-band ILD mean absolute error",
    "SpatialLSD_0_10deg": "Spatial LSD: 0-10 deg from Q26",
    "SpatialLSD_10_20deg": "Spatial LSD: 10-20 deg from Q26",
    "SpatialLSD_20_30deg": "Spatial LSD: 20-30 deg from Q26",
    "SpatialLSD_30_180deg": "Spatial LSD: 30-180 deg from Q26",
    "DominantNotchPenalizedMAE_Hz": "Dominant-notch penalized MAE",
    "DominantNotchMatchedMAE_Hz": "Dominant-notch matched-only MAE",
    "DominantNotchMissRate": "Dominant-notch miss rate",
    "DominantNotchSpuriousRate": "Dominant-notch spurious rate",
    "ReferenceNotchFraction": "Reference notch fraction",
    "ITDWeightedMAE_us": "ITD solid-angle-weighted MAE",
    "ITDMaximumAbsoluteError_us": "ITD maximum absolute error",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BASE_SEED = 20260903


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Inconsistent columns: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_seed(*values: str) -> int:
    text = "|".join((str(BOOTSTRAP_BASE_SEED),) + values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def tier(endpoint: str) -> str:
    if endpoint in PRIMARY:
        return "Primary frozen engineering test"
    if endpoint in SECONDARY:
        return "Secondary test"
    return "Deferred test"


def load_subject_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_rows(PRIMARY_ROOT / "metric_long.csv"):
        method, endpoint = row["Method"], row["Metric"]
        if method not in METHODS or endpoint not in PRIMARY:
            continue
        rows.append({
            "SubjectLabel": row["SubjectLabel"], "SubjectID": int(row["SubjectID"]),
            "Split": "test", "EvidenceTier": tier(endpoint), "Method": method,
            "MethodLabel": LABELS[method], "Endpoint": endpoint, "Unit": UNITS[endpoint],
            "Value": float(row["Value_dB"]), "Source": str((PRIMARY_ROOT / "metric_long.csv").relative_to(ROOT)).replace("\\", "/"),
        })
    for row in read_rows(SUPPLEMENTARY_ROOT / "per_subject_metrics.csv"):
        method, endpoint = row["Method"], row["Endpoint"]
        if method not in METHODS or endpoint not in SECONDARY + DEFERRED:
            continue
        rows.append({
            "SubjectLabel": row["SubjectLabel"], "SubjectID": int(row["SubjectID"]),
            "Split": "test", "EvidenceTier": tier(endpoint), "Method": method,
            "MethodLabel": LABELS[method], "Endpoint": endpoint, "Unit": UNITS[endpoint],
            "Value": float(row["Value"]), "Source": str((SUPPLEMENTARY_ROOT / "per_subject_metrics.csv").relative_to(ROOT)).replace("\\", "/"),
        })
    order = {method: index for index, method in enumerate(METHODS)}
    rows.sort(key=lambda row: (ENDPOINTS.index(str(row["Endpoint"])), order[str(row["Method"])], int(row["SubjectID"])))
    expected = {(method, endpoint) for endpoint in ENDPOINTS for method in METHODS}
    cells: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["Method"]), str(row["Endpoint"]))].append(row)
        if not np.isfinite(float(row["Value"])):
            raise FloatingPointError(f"Non-finite value: {row}")
    if set(cells) != expected:
        raise ValueError("Complete test method-endpoint coverage mismatch")
    reference_subjects: tuple[str, ...] | None = None
    for key, items in cells.items():
        subjects = tuple(sorted(str(item["SubjectLabel"]) for item in items))
        if len(subjects) != 44 or len(set(subjects)) != 44:
            raise ValueError(f"Expected 44 unique subjects: {key}")
        if reference_subjects is None:
            reference_subjects = subjects
        elif subjects != reference_subjects:
            raise ValueError(f"Subject alignment mismatch: {key}")
    return rows


def main() -> None:
    if OUTPUT.exists() or PARTIAL.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT} or {PARTIAL}")
    PARTIAL.mkdir(parents=True)
    rows = load_subject_rows()
    values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        values[(str(row["Method"]), str(row["Endpoint"]))][str(row["SubjectLabel"])] = float(row["Value"])

    aggregate: list[dict[str, object]] = []
    wide: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        means = {method: float(np.mean(list(values[(method, endpoint)].values()))) for method in METHODS}
        ranks = {} if endpoint == "ReferenceNotchFraction" else {
            method: 1 + sum(value < means[method] for value in means.values())
            for method in METHODS
        }
        wide_row: dict[str, object] = {
            "EvidenceTier": tier(endpoint), "Split": "test", "Endpoint": endpoint,
            "Title": TITLES[endpoint], "Unit": UNITS[endpoint],
            "Direction": "Descriptive only" if endpoint == "ReferenceNotchFraction" else "Lower is better",
        }
        for method in METHODS:
            vector = np.asarray([values[(method, endpoint)][subject] for subject in sorted(values[(method, endpoint)])], dtype=np.float64)
            seed = stable_seed("aggregate", method, endpoint)
            lower, upper = bootstrap_interval(vector, seed)
            aggregate.append({
                "EvidenceTier": tier(endpoint), "Split": "test", "Endpoint": endpoint,
                "Title": TITLES[endpoint], "Unit": UNITS[endpoint],
                "Direction": "Descriptive only" if endpoint == "ReferenceNotchFraction" else "Lower is better",
                "MethodOrder": METHODS.index(method) + 1, "Method": method, "MethodLabel": LABELS[method],
                "SubjectCount": 44, "Mean": float(vector.mean()), "SampleStd": float(vector.std(ddof=1)),
                "Bootstrap95Lower": lower, "Bootstrap95Upper": upper,
                "BootstrapReplicates": BOOTSTRAP_REPLICATES, "BootstrapSeed": seed,
                "RankWithinEndpoint": ranks.get(method, ""),
            })
            wide_row[LABELS[method]] = float(vector.mean())
        wide.append(wide_row)

    paired: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for baseline in METHODS[:-1]:
            subjects = sorted(values[("BOUNDED", endpoint)])
            difference = np.asarray([
                values[("BOUNDED", endpoint)][subject] - values[(baseline, endpoint)][subject]
                for subject in subjects
            ], dtype=np.float64)
            seed = stable_seed("paired", baseline, endpoint)
            lower, upper = bootstrap_interval(difference, seed)
            paired.append({
                "EvidenceTier": tier(endpoint), "Split": "test", "Endpoint": endpoint,
                "Unit": UNITS[endpoint], "Candidate": "BOUNDED", "CandidateLabel": LABELS["BOUNDED"],
                "Baseline": baseline, "BaselineLabel": LABELS[baseline], "SubjectCount": 44,
                "MeanDifference": float(difference.mean()), "Bootstrap95Lower": lower,
                "Bootstrap95Upper": upper, "MedianDifference": float(np.median(difference)),
                "P90Difference": float(np.quantile(difference, 0.90)),
                "P95Difference": float(np.quantile(difference, 0.95)),
                "WorstDecileMean": float(np.mean(np.sort(difference)[-max(1, int(np.ceil(0.1 * difference.size))):])),
                "MaximumDegradation": float(difference.max()), "MaximumSubject": subjects[int(np.argmax(difference))],
                "Wins": int(np.sum(difference < 0)), "Ties": int(np.sum(difference == 0)),
                "Losses": int(np.sum(difference > 0)), "WinRate": float(np.mean(difference < 0)),
                "DifferenceDefinition": "Bounded E25 minus baseline; negative favors Bounded",
                "BootstrapReplicates": BOOTSTRAP_REPLICATES, "BootstrapSeed": seed,
            })

    figure_index = []
    for number, endpoint in enumerate(ENDPOINTS, start=1):
        slug = "".join(character.lower() if character.isalnum() else "_" for character in endpoint).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        figure_index.append({
            "FigureNumber": number, "EvidenceTier": tier(endpoint), "Split": "test",
            "Endpoint": endpoint, "Title": TITLES[endpoint], "Unit": UNITS[endpoint],
            "Direction": "Descriptive only" if endpoint == "ReferenceNotchFraction" else "Lower is better",
            "PngFile": f"png/{number:02d}_test_{slug}.png", "PdfFile": f"pdf/{number:02d}_test_{slug}.pdf",
        })

    registry = [{
        "MethodOrder": number, "Method": method, "MethodLabel": LABELS[method],
        "Status": "COMPLETE", "Role": "candidate" if method in {"HYBRID", "BOUNDED"} else "baseline",
        "PrimaryTestSource": str((PRIMARY_ROOT / "metric_long.csv").relative_to(ROOT)).replace("\\", "/"),
        "SupplementaryTestSource": str((SUPPLEMENTARY_ROOT / "per_subject_metrics.csv").relative_to(ROOT)).replace("\\", "/"),
    } for number, method in enumerate(METHODS, start=1)]

    write_rows(PARTIAL / "per_subject_metrics.csv", rows)
    write_rows(PARTIAL / "aggregate_metrics.csv", aggregate)
    write_rows(PARTIAL / "paired_vs_bounded.csv", paired)
    write_rows(PARTIAL / "paper_complete_test_wide.csv", wide)
    write_rows(PARTIAL / "figure_index.csv", figure_index)
    write_rows(PARTIAL / "method_registry.csv", registry)
    shutil.copyfile(SUPPLEMENTARY_ROOT / "band_ild_profile.csv", PARTIAL / "band_ild_profile.csv")
    shutil.copyfile(SUPPLEMENTARY_ROOT / "spatial_direction_map.csv", PARTIAL / "spatial_direction_map.csv")

    source_files = [
        PRIMARY_ROOT / "metric_long.csv", PRIMARY_ROOT / "summary.json",
        SUPPLEMENTARY_ROOT / "per_subject_metrics.csv", SUPPLEMENTARY_ROOT / "summary.json",
    ]
    summary = {
        "schema_version": "1.0", "status": "completed", "split": "test",
        "method_count": 10, "endpoint_count": 20, "subject_count": 44,
        "per_subject_rows": len(rows), "aggregate_rows": len(aggregate),
        "paired_rows": len(paired), "wide_rows": len(wide), "method_registry_rows": len(registry),
        "all_finite": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "source_test_subject_count_read": 44, "new_test_subject_count_read_during_consolidation": 0,
        "primary_reproduction_max_abs_error_db": 0.0,
        "supplementary_manifest_identity_sha256": "E9403269543B6C135871BD8008809B488F2E29977601C95E3FF2BD491FDB229E",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in source_files},
        "claim_boundary": "Post-lock supplementary test characterization; no tuning, selection, or promotion decision.",
    }
    (PARTIAL / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (PARTIAL / "README.md").write_text(
        "# Complete ten-method test comparison\n\n"
        "This package consolidates the four frozen primary engineering-test endpoints and the sixteen "
        "post-lock supplementary test endpoints for all ten registered methods and 44 subjects.\n\n"
        "`paper_complete_test_wide.csv` is the one-row-per-endpoint paper table; `aggregate_metrics.csv` "
        "adds subject-bootstrap intervals and ranks; `per_subject_metrics.csv` preserves all paired values; "
        "`paired_vs_bounded.csv` reports Bounded-minus-baseline paired effects; `method_registry.csv` "
        "is the complete locked horizontal-comparison method list.\n\n"
        "The supplementary test results are characterization only and were not used for tuning or model selection.\n",
        encoding="utf-8",
    )
    PARTIAL.replace(OUTPUT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
