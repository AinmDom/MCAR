"""Combine five SONICOM baselines with the frozen three-method sparsity result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np


METHODS = (
    ("SHOnly", "SH only", "baseline"),
    ("SUpDEqSH", "SUpDEq + SH", "baseline"),
    ("SUpDEqNN", "SUpDEq + Natural Neighbor", "baseline"),
    ("SUpDEqBary", "SUpDEq + Barycentric", "baseline"),
    ("MCA", "MCA", "baseline"),
    ("MCARv32", "MCAR v3.2", "learned"),
    ("FSPAE", "FSP-AE", "learned"),
    ("RANF", "RANF", "learned"),
)
METHOD_IDS = tuple(item[0] for item in METHODS)
METHOD_LABEL = {item[0]: item[1] for item in METHODS}
METHOD_FAMILY = {item[0]: item[2] for item in METHODS}
METRICS = (
    "MeasuredDomainERB",
    "Contralateral25ERB",
    "ContralateralHemisphereHighFrequency",
    "HorizontalILDMAE",
)
COUNTS = (6, 14, 26)
FIELDS = (
    "SubjectLabel",
    "SubjectID",
    "DirectionCount",
    "Method",
    "MethodLabel",
    "Metric",
    "Value_dB",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("results/sonicom_five_baseline_sparsity_v1"),
    )
    parser.add_argument(
        "--learned-root",
        type=Path,
        default=Path("results/sonicom_learned_methods_sparsity_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/sonicom_eight_method_sparsity_v1"),
    )
    parser.add_argument("--bootstrap-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_812)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    if tuple(rows[0]) != FIELDS:
        raise ValueError(f"Unexpected metric schema in {path}: {tuple(rows[0])}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        method = row["Method"]
        output.append(
            {
                "SubjectLabel": row["SubjectLabel"],
                "SubjectID": int(row["SubjectID"]),
                "DirectionCount": int(row["DirectionCount"]),
                "Method": method,
                "MethodLabel": METHOD_LABEL[method],
                "Metric": row["Metric"],
                "Value_dB": float(row["Value_dB"]),
            }
        )
    return output


def validate_rows(rows: list[dict[str, object]]) -> None:
    expected = 44 * len(COUNTS) * len(METHOD_IDS) * len(METRICS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} combined rows, found {len(rows)}")
    keys: set[tuple[object, ...]] = set()
    for row in rows:
        key = (
            row["SubjectID"],
            row["DirectionCount"],
            row["Method"],
            row["Metric"],
        )
        if key in keys:
            raise ValueError(f"Duplicate metric key: {key}")
        keys.add(key)
        if (
            row["DirectionCount"] not in COUNTS
            or row["Method"] not in METHOD_IDS
            or row["Metric"] not in METRICS
            or not np.isfinite(row["Value_dB"])
        ):
            raise ValueError(f"Invalid metric row: {row}")
    subjects = {int(row["SubjectID"]) for row in rows}
    if len(subjects) != 44:
        raise ValueError(f"Expected 44 paired subjects, found {len(subjects)}")


def values_for(
    rows: list[dict[str, object]], method: str, metric: str, count: int
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (
            (int(row["SubjectID"]), float(row["Value_dB"]))
            for row in rows
            if row["Method"] == method
            and row["Metric"] == metric
            and row["DirectionCount"] == count
        ),
        key=lambda item: item[0],
    )
    return (
        np.asarray([item[0] for item in selected], dtype=np.int64),
        np.asarray([item[1] for item in selected], dtype=np.float64),
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in METHOD_IDS:
        for count in COUNTS:
            for metric in METRICS:
                subjects, values = values_for(rows, method, metric, count)
                if subjects.size != 44:
                    raise ValueError(f"Incomplete aggregate: {method}, {metric}, Q{count}")
                sd = float(values.std(ddof=1))
                output.append(
                    {
                        "Method": method,
                        "MethodLabel": METHOD_LABEL[method],
                        "MethodFamily": METHOD_FAMILY[method],
                        "DirectionCount": count,
                        "Metric": metric,
                        "SubjectCount": int(subjects.size),
                        "Mean_dB": float(values.mean()),
                        "SD_dB": sd,
                        "CI95HalfWidth_dB": float(1.96 * sd / np.sqrt(subjects.size)),
                    }
                )
    return output


def robustness_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    x = np.log2(26 / np.asarray(COUNTS, dtype=np.float64))
    output: list[dict[str, object]] = []
    for method in METHOD_IDS:
        for metric in METRICS:
            subject_reference: np.ndarray | None = None
            matrix = []
            for count in COUNTS:
                subjects, values = values_for(rows, method, metric, count)
                if subject_reference is None:
                    subject_reference = subjects
                elif not np.array_equal(subject_reference, subjects):
                    raise ValueError(f"Subject pairing failed: {method}, {metric}")
                matrix.append(values)
            values = np.column_stack(matrix)
            degradation = values[:, 0] - values[:, -1]
            slopes = np.asarray(
                [np.polyfit(x, listener_values, 1)[0] for listener_values in values]
            )
            output.append(
                {
                    "Method": method,
                    "MethodLabel": METHOD_LABEL[method],
                    "MethodFamily": METHOD_FAMILY[method],
                    "Metric": metric,
                    "Q6Mean_dB": float(values[:, 0].mean()),
                    "Q26Mean_dB": float(values[:, -1].mean()),
                    "Q6MinusQ26Mean_dB": float(degradation.mean()),
                    "Q6MinusQ26SD_dB": float(degradation.std(ddof=1)),
                    "SlopePerHalvingMean_dB": float(slopes.mean()),
                    "SlopePerHalvingSD_dB": float(slopes.std(ddof=1)),
                }
            )
    return output


def bootstrap_interval(
    difference: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float, float]:
    indices = rng.integers(0, difference.size, size=(replicates, difference.size))
    means = difference[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    p_value = min(
        1.0,
        2
        * min(
            (np.count_nonzero(means <= 0) + 1) / (replicates + 1),
            (np.count_nonzero(means >= 0) + 1) / (replicates + 1),
        ),
    )
    return float(lower), float(upper), float(p_value)


def pairwise_rows(
    rows: list[dict[str, object]], seed: int, replicates: int
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for metric in METRICS:
        for first_index, method_a in enumerate(METHOD_IDS):
            for method_b in METHOD_IDS[first_index + 1 :]:
                subjects_a6, a6 = values_for(rows, method_a, metric, 6)
                subjects_a26, a26 = values_for(rows, method_a, metric, 26)
                subjects_b6, b6 = values_for(rows, method_b, metric, 6)
                subjects_b26, b26 = values_for(rows, method_b, metric, 26)
                if not (
                    np.array_equal(subjects_a6, subjects_a26)
                    and np.array_equal(subjects_a6, subjects_b6)
                    and np.array_equal(subjects_a6, subjects_b26)
                ):
                    raise ValueError(f"Pairing failed: {method_a}, {method_b}, {metric}")
                for contrast, difference in (
                    ("Q6Endpoint", a6 - b6),
                    ("Q6MinusQ26Degradation", (a6 - a26) - (b6 - b26)),
                ):
                    lower, upper, p_value = bootstrap_interval(
                        difference, rng, replicates
                    )
                    output.append(
                        {
                            "Metric": metric,
                            "Contrast": contrast,
                            "MethodA": method_a,
                            "MethodB": method_b,
                            "MethodAFamily": METHOD_FAMILY[method_a],
                            "MethodBFamily": METHOD_FAMILY[method_b],
                            "MeanDifferenceAminusB_dB": float(difference.mean()),
                            "CI95Lower_dB": lower,
                            "CI95Upper_dB": upper,
                            "TwoSidedBootstrapP": p_value,
                            "BootstrapReplicates": replicates,
                            "Seed": seed,
                        }
                    )
    return output


def ranking_rows(
    aggregate: list[dict[str, object]], robustness: list[dict[str, object]]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric in METRICS:
        for count in COUNTS:
            selected = [
                row
                for row in aggregate
                if row["Metric"] == metric and row["DirectionCount"] == count
            ]
            for rank, row in enumerate(sorted(selected, key=lambda item: item["Mean_dB"]), 1):
                output.append(
                    {
                        "Metric": metric,
                        "Quantity": f"Q{count}Endpoint",
                        "Rank": rank,
                        "Method": row["Method"],
                        "MethodLabel": row["MethodLabel"],
                        "Value_dB": row["Mean_dB"],
                    }
                )
        selected = [row for row in robustness if row["Metric"] == metric]
        for rank, row in enumerate(
            sorted(selected, key=lambda item: item["Q6MinusQ26Mean_dB"]), 1
        ):
            output.append(
                {
                    "Metric": metric,
                    "Quantity": "Q6MinusQ26Degradation",
                    "Rank": rank,
                    "Method": row["Method"],
                    "MethodLabel": row["MethodLabel"],
                    "Value_dB": row["Q6MinusQ26Mean_dB"],
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(aggregate: list[dict[str, object]], output_root: Path) -> None:
    """Write a dependency-free SVG so the frozen analysis needs only NumPy."""
    colors = (
        "#4c78a8",
        "#f58518",
        "#e45756",
        "#72b7b2",
        "#54a24b",
        "#eeca3b",
        "#b279a2",
        "#ff9da6",
    )
    width, height = 1400, 1000
    panel_width = 650
    origins = ((75, 70), (735, 70), (75, 535), (735, 535))
    plot_left, plot_top, plot_width, plot_height = 72, 48, 545, 280
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:18px;font-weight:600}.tick{font-size:13px}.axis{stroke:#333;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:12px}</style>',
    ]
    log_min, log_max = math.log(COUNTS[0]), math.log(COUNTS[-1])
    for metric, (origin_x, origin_y) in zip(METRICS, origins):
        selected_metric = [row for row in aggregate if row["Metric"] == metric]
        lower = min(
            float(row["Mean_dB"]) - float(row["CI95HalfWidth_dB"])
            for row in selected_metric
        )
        upper = max(
            float(row["Mean_dB"]) + float(row["CI95HalfWidth_dB"])
            for row in selected_metric
        )
        padding = max((upper - lower) * 0.08, 0.02)
        y_min, y_max = lower - padding, upper + padding

        def x_coord(count: int) -> float:
            return origin_x + plot_left + plot_width * (
                (math.log(count) - log_min) / (log_max - log_min)
            )

        def y_coord(value: float) -> float:
            return origin_y + plot_top + plot_height * (
                1.0 - (value - y_min) / (y_max - y_min)
            )

        x0, y0 = origin_x + plot_left, origin_y + plot_top
        svg.append(
            f'<text class="title" x="{origin_x + panel_width / 2:.1f}" y="{origin_y + 20}" text-anchor="middle">{html.escape(metric)}</text>'
        )
        for tick_index in range(5):
            value = y_min + tick_index * (y_max - y_min) / 4
            y = y_coord(value)
            svg.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_width}" y2="{y:.2f}"/>')
            svg.append(f'<text class="tick" x="{x0 - 9}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>')
        for count in COUNTS:
            x = x_coord(count)
            svg.append(f'<line class="grid" x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + plot_height}"/>')
            svg.append(f'<text class="tick" x="{x:.2f}" y="{y0 + plot_height + 22}" text-anchor="middle">{count}</text>')
        svg.extend(
            [
                f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}"/>',
                f'<line class="axis" x1="{x0}" y1="{y0 + plot_height}" x2="{x0 + plot_width}" y2="{y0 + plot_height}"/>',
                f'<text class="tick" x="{x0 + plot_width / 2:.1f}" y="{y0 + plot_height + 46}" text-anchor="middle">Observed directions Q</text>',
                f'<text class="tick" transform="translate({origin_x + 18},{y0 + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">Error (dB)</text>',
            ]
        )
        for method_index, method in enumerate(METHOD_IDS):
            selected = sorted(
                (row for row in selected_metric if row["Method"] == method),
                key=lambda row: row["DirectionCount"],
            )
            points = [
                (x_coord(int(row["DirectionCount"])), y_coord(float(row["Mean_dB"])))
                for row in selected
            ]
            color = colors[method_index]
            svg.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="'
                + " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                + '"/>'
            )
            for row, (x, y) in zip(selected, points):
                ci = float(row["CI95HalfWidth_dB"])
                y_low = y_coord(float(row["Mean_dB"]) - ci)
                y_high = y_coord(float(row["Mean_dB"]) + ci)
                svg.extend(
                    [
                        f'<line stroke="{color}" x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}"/>',
                        f'<line stroke="{color}" x1="{x - 4:.2f}" y1="{y_low:.2f}" x2="{x + 4:.2f}" y2="{y_low:.2f}"/>',
                        f'<line stroke="{color}" x1="{x - 4:.2f}" y1="{y_high:.2f}" x2="{x + 4:.2f}" y2="{y_high:.2f}"/>',
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>',
                    ]
                )
    legend_y = 975
    for index, method in enumerate(METHOD_IDS):
        x = 70 + index * 165
        svg.append(f'<line x1="{x}" y1="{legend_y - 5}" x2="{x + 24}" y2="{legend_y - 5}" stroke="{colors[index]}" stroke-width="3"/>')
        svg.append(f'<circle cx="{x + 12}" cy="{legend_y - 5}" r="4" fill="{colors[index]}"/>')
        svg.append(f'<text class="legend" x="{x + 30}" y="{legend_y}">{html.escape(METHOD_LABEL[method])}</text>')
    svg.append("</svg>")
    (output_root / "figures" / "eight_method_sparsity_curves.svg").write_text(
        "\n".join(svg) + "\n", encoding="utf-8"
    )


def write_readme(
    output_root: Path,
    robustness: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    learned_vs_baseline: list[dict[str, object]],
) -> None:
    by_robustness = {
        (row["Method"], row["Metric"]): row for row in robustness
    }

    def contrast(method_a: str, method_b: str, metric: str) -> dict[str, object]:
        for row in pairwise:
            if row["Metric"] != metric or row["Contrast"] != "Q6Endpoint":
                continue
            if row["MethodA"] == method_a and row["MethodB"] == method_b:
                return row
            if row["MethodA"] == method_b and row["MethodB"] == method_a:
                return {
                    **row,
                    "MeanDifferenceAminusB_dB": -float(
                        row["MeanDifferenceAminusB_dB"]
                    ),
                    "CI95Lower_dB": -float(row["CI95Upper_dB"]),
                    "CI95Upper_dB": -float(row["CI95Lower_dB"]),
                }
        raise ValueError(f"Missing Q6 contrast: {method_a}, {method_b}, {metric}")

    strongest_baseline = {}
    for metric in METRICS:
        strongest_baseline[metric] = min(
            METHOD_IDS[:5],
            key=lambda method: by_robustness[(method, metric)]["Q6Mean_dB"],
        )
    lines = [
        "# SONICOM eight-method sparsity experiment",
        "",
        "Five horizontal baselines are combined with the unchanged MCAR v3.2, FSP-AE, and RANF listener-level results.",
        "All methods use the same 44 test listeners, Q6/Q14/Q26 grids, and 767-target mask.",
        "",
        "## Main findings",
        "",
        "- RANF ranks first on all four Q6 endpoints. Its Q6 means are "
        + " / ".join(
            f"{by_robustness[('RANF', metric)]['Q6Mean_dB']:.3f}"
            for metric in METRICS
        )
        + " dB in the metric order used below.",
        "- MCAR v3.2 is significantly better than the strongest Q6 horizontal baseline for every metric. MCAR-minus-baseline differences are "
        + " / ".join(
            f"{contrast('MCARv32', strongest_baseline[metric], metric)['MeanDifferenceAminusB_dB']:.3f}"
            for metric in METRICS
        )
        + " dB (all paired-bootstrap p=0.0002).",
        "- FSP-AE beats the strongest Q6 baseline only for contralateral-hemisphere high-frequency error; its FSP-AE-minus-baseline differences are "
        + " / ".join(
            f"{contrast('FSPAE', strongest_baseline[metric], metric)['MeanDifferenceAminusB_dB']:.3f}"
            for metric in METRICS
        )
        + " dB.",
        "- RANF also degrades less from Q26 to Q6 than MCAR on all four metrics. A flat or negative curve is not treated as robust by itself when its absolute endpoint error remains high (notably SH only).",
        "",
        "## Robustness summary",
        "",
        "| Method | Metric | Q6 | Q26 | Q6-Q26 | Slope / halving |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in robustness:
        lines.append(
            "| {MethodLabel} | {Metric} | {Q6Mean_dB:.3f} | {Q26Mean_dB:.3f} | "
            "{Q6MinusQ26Mean_dB:.3f} | {SlopePerHalvingMean_dB:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            f"`paired_bootstrap_all_pairs.csv` contains 224 contrasts. `paired_bootstrap_learned_vs_baselines.csv` contains the {len(learned_vs_baseline)} learned-versus-baseline contrasts.",
            "Differences are Method A minus Method B; negative favors Method A.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    baseline_path = args.baseline_root / "metric_long.csv"
    learned_path = args.learned_root / "metric_long.csv"
    baseline_raw = read_rows(baseline_path)
    learned_raw = read_rows(learned_path)
    if {row["Method"] for row in baseline_raw} != set(METHOD_IDS[:5]):
        raise ValueError("The baseline table does not contain the frozen five methods")
    if {row["Method"] for row in learned_raw} != set(METHOD_IDS[5:]):
        raise ValueError("The learned table does not contain the frozen three methods")
    baseline = normalized_rows(baseline_raw)
    learned = normalized_rows(learned_raw)
    rows = baseline + learned
    validate_rows(rows)
    aggregate = aggregate_rows(rows)
    robustness = robustness_rows(rows)
    pairwise = pairwise_rows(rows, args.seed, args.bootstrap_count)
    learned_vs_baseline = [
        row
        for row in pairwise
        if {row["MethodAFamily"], row["MethodBFamily"]} == {"learned", "baseline"}
    ]
    rankings = ranking_rows(aggregate, robustness)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "figures").mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "metric_long.csv", rows)
    write_csv(args.output_root / "aggregate_metrics.csv", aggregate)
    write_csv(args.output_root / "robustness_summary.csv", robustness)
    write_csv(args.output_root / "paired_bootstrap_all_pairs.csv", pairwise)
    write_csv(
        args.output_root / "paired_bootstrap_learned_vs_baselines.csv",
        learned_vs_baseline,
    )
    write_csv(args.output_root / "rankings.csv", rankings)
    write_plot(aggregate, args.output_root)
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "subject_count": 44,
        "direction_counts": list(COUNTS),
        "method_count": len(METHOD_IDS),
        "metric_row_count": len(rows),
        "all_pair_contrast_count": len(pairwise),
        "learned_vs_baseline_contrast_count": len(learned_vs_baseline),
        "fixed_evaluation_direction_count": 767,
        "bootstrap_replicates": args.bootstrap_count,
        "bootstrap_seed": args.seed,
        "source_tables": {
            "five_baselines": str(baseline_path),
            "five_baselines_sha256": sha256(baseline_path),
            "three_learned_methods": str(learned_path),
            "three_learned_methods_sha256": sha256(learned_path),
            "learned_rows_reused_without_recomputation": True,
        },
        "aggregate": aggregate,
        "robustness": robustness,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(args.output_root, robustness, pairwise, learned_vs_baseline)
    print(f"SONICOM eight-method sparsity analysis complete: {args.output_root}")


if __name__ == "__main__":
    main()
