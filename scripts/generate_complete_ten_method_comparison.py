"""Build the registered ten-method horizontal comparison from frozen result CSVs.

This script is consolidation-only.  It reads no raw HRTF/test data and runs no model.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "sonicom_complete_ten_method_comparison_v1"
METHODS = (
    "SHOnly",
    "SUpDEqSH",
    "SUpDEqNN",
    "SUpDEqBary",
    "MCA",
    "MCARv351",
    "FSPAE",
    "RANF",
    "HYBRID",
    "BOUNDED",
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
PRIMARY_ENDPOINTS = (
    "FullSphereERB",
    "Contralateral25ERB",
    "ContralateralHighFrequency",
    "HorizontalILDMAE",
)
SECONDARY_ENDPOINTS = (
    "FullSphereLSD",
    "HFFirstDifferenceMAE",
    "HFSecondDifferenceMAE",
    "MultiScaleNotchDepthMAE",
    "ERBBandILDMean",
    "SpatialLSD_0_10deg",
    "SpatialLSD_10_20deg",
    "SpatialLSD_20_30deg",
    "SpatialLSD_30_180deg",
)
DEFERRED_ENDPOINTS = (
    "DominantNotchPenalizedMAE_Hz",
    "DominantNotchMatchedMAE_Hz",
    "DominantNotchMissRate",
    "DominantNotchSpuriousRate",
    "ReferenceNotchFraction",
    "ITDWeightedMAE_us",
    "ITDMaximumAbsoluteError_us",
)
UNITS = {
    **{endpoint: "dB" for endpoint in PRIMARY_ENDPOINTS},
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


def read_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Inconsistent columns: {name}")
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def canonical_subject_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validation: list[dict[str, object]] = []
    sources = (
        (
            "results/sonicom_fsp_ae_q26_formal_validation/metric_long.csv",
            {method: method for method in METHODS[:5]},
        ),
        (
            "results/sonicom_bounded_mcar_film_correction_final_e25_validation/metric_long.csv",
            {"MCAR": "MCARv351", "PARENT": "HYBRID", "HYBRID": "BOUNDED"},
        ),
        (
            "results/sonicom_film_siren_gl_final_d1d2_notch_vs_ranf_fsp_v351_validation/metric_long.csv",
            {"FSPAE": "FSPAE", "RANF": "RANF"},
        ),
    )
    for source, mapping in sources:
        for row in read_rows(source):
            if row["Method"] not in mapping:
                continue
            method = mapping[row["Method"]]
            validation.append(
                {
                    "EvidenceTier": "Primary validation",
                    "Split": "val",
                    "SubjectLabel": row["SubjectLabel"],
                    "SubjectID": int(row["SubjectID"]),
                    "Method": method,
                    "MethodLabel": LABELS[method],
                    "Endpoint": row["Metric"],
                    "Unit": "dB",
                    "Value": float(row["Value_dB"]),
                    "Source": source,
                }
            )
    test_source = (
        "results/sonicom_bounded_mcar_film_correction_final_e25_"
        "frozen_test_nine_method/metric_long.csv"
    )
    test = []
    for row in read_rows(test_source):
        if row["Method"] not in METHODS:
            continue
        method = row["Method"]
        test.append(
            {
                "EvidenceTier": "Primary frozen engineering test",
                "Split": "test",
                "SubjectLabel": row["SubjectLabel"],
                "SubjectID": int(row["SubjectID"]),
                "Method": method,
                "MethodLabel": LABELS[method],
                "Endpoint": row["Metric"],
                "Unit": "dB",
                "Value": float(row["Value_dB"]),
                "Source": test_source,
            }
        )
    order = {method: index for index, method in enumerate(METHODS)}
    key = lambda row: (order[row["Method"]], PRIMARY_ENDPOINTS.index(row["Endpoint"]), row["SubjectID"])
    return sorted(validation, key=key), sorted(test, key=key)


def validate_subject_rows(rows: list[dict[str, object]], methods: tuple[str, ...]) -> None:
    by_cell: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        value = float(row["Value"])
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite value in {row['Method']}/{row['Endpoint']}")
        by_cell[(str(row["Method"]), str(row["Endpoint"]))].append(row)
    if set(by_cell) != {(method, endpoint) for method in methods for endpoint in PRIMARY_ENDPOINTS}:
        raise ValueError("Primary method-endpoint coverage mismatch")
    reference_subjects: tuple[str, ...] | None = None
    for cell, values in by_cell.items():
        subjects = tuple(sorted(str(row["SubjectLabel"]) for row in values))
        if len(subjects) != 44 or len(set(subjects)) != 44:
            raise ValueError(f"Expected 44 unique subjects: {cell}")
        if reference_subjects is None:
            reference_subjects = subjects
        elif subjects != reference_subjects:
            raise ValueError(f"Subject alignment mismatch: {cell}")


def aggregate_subject_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(str(row["Method"]), str(row["Endpoint"]))].append(float(row["Value"]))
    output = {}
    for key, items in values.items():
        vector = np.asarray(items, dtype=np.float64)
        output[key] = {
            "SubjectCount": int(vector.size),
            "Mean": float(vector.mean()),
            "SampleStd": float(vector.std(ddof=1)),
        }
    return output


def supplementary_means() -> tuple[dict[tuple[str, str], dict[str, float]], list[dict[str, object]], list[dict[str, object]]]:
    output: dict[tuple[str, str], dict[str, float]] = {}
    secondary_subjects: list[dict[str, object]] = []
    deferred_subjects: list[dict[str, object]] = []
    method_maps = (
        (
            "results/sonicom_film_secondary_metrics_v1_validation",
            {"BOUNDED": "BOUNDED", "HYBRID": "HYBRID", "MCAR": "MCARv351"},
        ),
        (
            "results/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation",
            {"RANF": "RANF", "FSPAE": "FSPAE"},
        ),
    )
    for root, mapping in method_maps:
        for row in read_rows(f"{root}/per_subject_metrics.csv"):
            if row["Method"] not in mapping:
                continue
            method = mapping[row["Method"]]
            secondary_subjects.append(
                {
                    "SubjectLabel": row["SubjectLabel"],
                    "SubjectID": int(row["SubjectID"]),
                    "Method": method,
                    "Endpoint": row["Endpoint"],
                    "Value": float(row["Value"]),
                }
            )
        for row in read_rows(f"{root}/spatial_distance_bins.csv"):
            if row["RecordType"] != "subject" or row["Method"] not in mapping:
                continue
            method = mapping[row["Method"]]
            secondary_subjects.append(
                {
                    "SubjectLabel": row["SubjectLabel"],
                    "SubjectID": int(row["SubjectID"]),
                    "Method": method,
                    "Endpoint": f"SpatialLSD_{row['DistanceBin_deg']}deg",
                    "Value": float(row["LSD_dB"]),
                }
            )
        for row in read_rows(f"{root}/per_subject_endpoints.csv") if "baseline_extension" in root else read_rows("results/sonicom_film_deferred_secondary_metrics_v1_validation/per_subject_endpoints.csv"):
            if row["Method"] not in mapping:
                continue
            method = mapping[row["Method"]]
            deferred_subjects.append(
                {
                    "SubjectLabel": row["SubjectLabel"],
                    "SubjectID": int(row["SubjectID"]),
                    "Method": method,
                    "Endpoint": row["Endpoint"],
                    "Value": float(row["Value"]),
                }
            )
    for tier_rows, endpoints in ((secondary_subjects, SECONDARY_ENDPOINTS), (deferred_subjects, DEFERRED_ENDPOINTS)):
        cells: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in tier_rows:
            cells[(str(row["Method"]), str(row["Endpoint"]))].append(float(row["Value"]))
        expected_methods = ("MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED")
        if set(cells) != {(method, endpoint) for method in expected_methods for endpoint in endpoints}:
            missing = {(method, endpoint) for method in expected_methods for endpoint in endpoints} - set(cells)
            extra = set(cells) - {(method, endpoint) for method in expected_methods for endpoint in endpoints}
            raise ValueError(f"Supplementary coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        for key, items in cells.items():
            vector = np.asarray(items, dtype=np.float64)
            if vector.size != 44 or not np.all(np.isfinite(vector)):
                raise ValueError(f"Invalid supplementary cell: {key}/{vector.size}")
            output[key] = {
                "SubjectCount": 44,
                "Mean": float(vector.mean()),
                "SampleStd": float(vector.std(ddof=1)),
            }
    return output, secondary_subjects, deferred_subjects


def paired_rows(
    tier: str,
    split: str,
    subject_rows: list[dict[str, object]],
    endpoints: tuple[str, ...],
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in subject_rows:
        values[(str(row["Method"]), str(row["Endpoint"]))][str(row["SubjectLabel"])] = float(row["Value"])
    result = []
    for method in methods:
        if method == "BOUNDED":
            continue
        for endpoint in endpoints:
            candidate = values[("BOUNDED", endpoint)]
            baseline = values[(method, endpoint)]
            if set(candidate) != set(baseline) or len(candidate) != 44:
                raise ValueError(f"Paired alignment mismatch: {tier}/{method}/{endpoint}")
            subjects = sorted(candidate)
            difference = np.asarray([candidate[s] - baseline[s] for s in subjects], dtype=np.float64)
            seed = 20260830 + 1000 * list(METHODS).index(method) + list(endpoints).index(endpoint)
            rng = np.random.Generator(np.random.PCG64(seed))
            draws = rng.integers(0, difference.size, size=(10000, difference.size))
            boot = difference[draws].mean(axis=1)
            result.append(
                {
                    "EvidenceTier": tier,
                    "Split": split,
                    "Endpoint": endpoint,
                    "Unit": UNITS[endpoint],
                    "Candidate": "BOUNDED",
                    "CandidateLabel": LABELS["BOUNDED"],
                    "Baseline": method,
                    "BaselineLabel": LABELS[method],
                    "SubjectCount": 44,
                    "MeanDifference": float(difference.mean()),
                    "Bootstrap95Lower": float(np.quantile(boot, 0.025)),
                    "Bootstrap95Upper": float(np.quantile(boot, 0.975)),
                    "Wins": int(np.sum(difference < 0)),
                    "Ties": int(np.sum(difference == 0)),
                    "Losses": int(np.sum(difference > 0)),
                    "DifferenceDefinition": "Bounded E25 minus baseline; lower is better",
                    "BootstrapReplicates": 10000,
                    "BootstrapSeed": seed,
                }
            )
    return result


def long_mean_rows(
    validation: dict[tuple[str, str], dict[str, float]],
    test: dict[tuple[str, str], dict[str, float]],
    supplementary: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, object]]:
    tiers = (
        ("Primary validation", "val", PRIMARY_ENDPOINTS, validation, set(METHODS), "Validation primary"),
        ("Primary frozen engineering test", "test", PRIMARY_ENDPOINTS, test, set(METHODS) - {"HYBRID"}, "Historical frozen engineering test"),
        ("Secondary validation", "val", SECONDARY_ENDPOINTS, supplementary, {"MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED"}, "Validation supplementary"),
        ("Deferred validation", "val", DEFERRED_ENDPOINTS, supplementary, {"MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED"}, "Exploratory validation"),
    )
    rows = []
    for tier, split, endpoints, values, available, evidence in tiers:
        for endpoint in endpoints:
            available_means = {m: values[(m, endpoint)]["Mean"] for m in available}
            ordered = sorted(available_means, key=lambda m: (available_means[m], METHODS.index(m)))
            ranks = {} if endpoint == "ReferenceNotchFraction" else {method: index + 1 for index, method in enumerate(ordered)}
            for method in METHODS:
                is_available = method in available
                item = values.get((method, endpoint), {})
                reason = ""
                if not is_available:
                    reason = "Hybrid E190 was not included in the frozen test method set" if tier.startswith("Primary frozen") else "No frozen same-protocol reconstruction/result for this endpoint tier"
                rows.append(
                    {
                        "EvidenceTier": tier,
                        "Split": split,
                        "EvidenceLevel": evidence,
                        "Endpoint": endpoint,
                        "Unit": UNITS[endpoint],
                        "Direction": "Descriptive only" if endpoint == "ReferenceNotchFraction" else "Lower is better",
                        "MethodOrder": METHODS.index(method) + 1,
                        "Method": method,
                        "MethodLabel": LABELS[method],
                        "Status": "AVAILABLE" if is_available else "NOT RUN",
                        "Reason": reason,
                        "SubjectCount": item.get("SubjectCount", ""),
                        "Mean": item.get("Mean", ""),
                        "SampleStd": item.get("SampleStd", ""),
                        "RankWithinTier": ranks.get(method, ""),
                    }
                )
    return rows


def wide_rows(mean_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in mean_rows:
        grouped[(str(row["EvidenceTier"]), str(row["Split"]), str(row["EvidenceLevel"]), str(row["Endpoint"]))][str(row["Method"])] = row
    output = []
    for (tier, split, evidence, endpoint), items in grouped.items():
        row: dict[str, object] = {
            "EvidenceTier": tier,
            "Split": split,
            "EvidenceLevel": evidence,
            "Endpoint": endpoint,
            "Unit": UNITS[endpoint],
            "Direction": "Descriptive only" if endpoint == "ReferenceNotchFraction" else "Lower is better",
        }
        for method in METHODS:
            item = items[method]
            row[LABELS[method]] = item["Mean"] if item["Status"] == "AVAILABLE" else "NOT RUN"
        output.append(row)
    return output


def availability_rows() -> list[dict[str, object]]:
    available_secondary = {"MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED"}
    groups = (
        ("Primary validation", set(METHODS), "AVAILABLE", "44-subject strict validation"),
        ("Primary frozen engineering test", set(METHODS) - {"HYBRID"}, "NOT RUN", "Hybrid E190 was not in the frozen test set"),
        ("Secondary validation", available_secondary, "NOT RUN", "No frozen same-protocol secondary reconstruction/result"),
        ("Deferred validation", available_secondary, "NOT RUN", "No frozen same-protocol exploratory reconstruction/result"),
        ("Mechanism validation", {"BOUNDED"}, "NOT APPLICABLE", "No common bounded-gate/correction internal quantity"),
        ("Efficiency validation", {"BOUNDED"}, "NOT AVAILABLE", "No frozen same-hardware benchmark"),
        ("Model-based localization", set(), "NOT RUN", "Frozen official dependency unavailable"),
    )
    rows = []
    for method in METHODS:
        for group, available, missing_status, missing_reason in groups:
            status = "AVAILABLE" if method in available else missing_status
            reason = "Existing frozen evidence" if status == "AVAILABLE" else missing_reason
            rows.append(
                {
                    "MethodOrder": METHODS.index(method) + 1,
                    "Method": method,
                    "MethodLabel": LABELS[method],
                    "MetricGroup": group,
                    "Status": status,
                    "Reason": reason,
                }
            )
    return rows


def profile_rows(filename: str) -> list[dict[str, object]]:
    rows = []
    sources = (
        ("results/sonicom_film_secondary_metrics_v1_validation", {"BOUNDED": "BOUNDED", "HYBRID": "HYBRID", "MCAR": "MCARv351"}),
        ("results/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation", {"RANF": "RANF", "FSPAE": "FSPAE"}),
    )
    for root, mapping in sources:
        for row in read_rows(f"{root}/{filename}"):
            if row.get("RecordType", "aggregate") != "aggregate" or row["Method"] not in mapping:
                continue
            method = mapping[row["Method"]]
            clean: dict[str, object] = {"Method": method, "MethodLabel": LABELS[method]}
            clean.update({key: value for key, value in row.items() if key not in {"RecordType", "SubjectLabel", "SubjectID", "Method"}})
            rows.append(clean)
    if filename == "band_ild_profile.csv":
        rows.sort(key=lambda row: (METHODS.index(str(row["Method"])), float(row["BandIndex"])))
    else:
        rows.sort(key=lambda row: (METHODS.index(str(row["Method"])), float(row["DirectionIndexZeroBased"])))
    return rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    validation_rows, test_rows = canonical_subject_rows()
    validate_subject_rows(validation_rows, METHODS)
    validate_subject_rows(test_rows, tuple(method for method in METHODS if method != "HYBRID"))
    validation_means = aggregate_subject_rows(validation_rows)
    test_means = aggregate_subject_rows(test_rows)
    supplementary, secondary_subjects, deferred_subjects = supplementary_means()
    means = long_mean_rows(validation_means, test_means, supplementary)
    pairs = []
    pairs.extend(paired_rows("Primary validation", "val", validation_rows, PRIMARY_ENDPOINTS, METHODS))
    pairs.extend(paired_rows("Primary frozen engineering test", "test", test_rows, PRIMARY_ENDPOINTS, tuple(m for m in METHODS if m != "HYBRID")))
    pairs.extend(paired_rows("Secondary validation", "val", secondary_subjects, SECONDARY_ENDPOINTS, ("MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED")))
    paired_deferred_endpoints = tuple(e for e in DEFERRED_ENDPOINTS if e != "ReferenceNotchFraction")
    pairs.extend(paired_rows("Deferred validation", "val", deferred_subjects, paired_deferred_endpoints, ("MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED")))

    registry = json.loads((ROOT / "configs/experiments/sonicom_complete_horizontal_comparison_methods_v1.json").read_text(encoding="utf-8"))
    registry_rows = [
        {
            "MethodOrder": item["order"],
            "Method": item["id"],
            "MethodLabel": item["paper_label"],
            "Family": item["family"],
            "RegisteredStatus": registry["status"].upper(),
        }
        for item in registry["methods"]
    ]
    write_rows("method_registry.csv", registry_rows)
    write_rows("method_endpoint_means.csv", means)
    write_rows("paper_complete_comparison_wide.csv", wide_rows(means))
    write_rows("paired_vs_bounded.csv", pairs)
    write_rows("availability_matrix.csv", availability_rows())
    write_rows("band_ild_profile_aggregate.csv", profile_rows("band_ild_profile.csv"))
    direction_rows = profile_rows("spatial_direction_map.csv")
    write_rows("spatial_direction_map.csv", direction_rows)
    efficiency = read_rows("results/sonicom_bounded_mcar_film_efficiency_v1_validation/efficiency_summary.csv")
    write_rows("bounded_efficiency_summary.csv", efficiency)
    mechanism = [
        {"Method": "BOUNDED", "MethodLabel": LABELS["BOUNDED"], **row}
        for row in read_rows("results/sonicom_film_deferred_secondary_metrics_v1_validation/mechanism_aggregate.csv")
    ]
    correction_correlation = [
        {"Method": "BOUNDED", "MethodLabel": LABELS["BOUNDED"], **row}
        for row in read_rows("results/sonicom_film_deferred_secondary_metrics_v1_validation/correction_benefit_spearman.csv")
    ]
    write_rows("bounded_mechanism_aggregate.csv", mechanism)
    write_rows("bounded_correction_benefit_spearman.csv", correction_correlation)

    numeric_means = [float(row["Mean"]) for row in means if row["Status"] == "AVAILABLE"]
    if len(means) != 240 or len(numeric_means) != 156 or not np.all(np.isfinite(numeric_means)):
        raise ValueError("Mean-table completeness check failed")
    if len(pairs) != 9 * 4 + 8 * 4 + 4 * 9 + 4 * 6:
        raise ValueError("Paired-table row count mismatch")
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "registered_methods": list(METHODS),
        "method_count": 10,
        "primary_validation_method_count": 10,
        "primary_test_method_count": 9,
        "primary_test_missing": ["HYBRID"],
        "secondary_validation_method_count": 5,
        "deferred_validation_method_count": 5,
        "method_endpoint_rows": len(means),
        "available_numeric_method_endpoint_rows": len(numeric_means),
        "paired_rows": len(pairs),
        "band_profile_rows": len(profile_rows("band_ild_profile.csv")),
        "spatial_direction_rows": len(direction_rows),
        "bounded_mechanism_rows": len(mechanism),
        "bounded_correction_correlation_rows": len(correction_correlation),
        "all_available_means_finite": True,
        "new_test_subject_count_read": 0,
        "upstream_frozen_test_subject_count": 44,
        "claim_boundary": "Validation, historical frozen engineering test, supplementary validation, and exploratory validation remain separate. Missing cells are not imputed.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Complete ten-method horizontal comparison",
        "",
        "The canonical comparison set is frozen in the user-specified order. The single wide table is `paper_complete_comparison_wide.csv`; numeric cells are means across 44 subjects and `NOT RUN` is structural missingness.",
        "",
        "- Primary validation: 10/10 methods, four strict endpoints.",
        "- Historical frozen engineering test: 9/10 methods; Hybrid E190 was not run.",
        "- Secondary and deferred validation: 5/10 methods with frozen same-protocol evidence.",
        "- No raw test data or new model inference was used in this consolidation.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
