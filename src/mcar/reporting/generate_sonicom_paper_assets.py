"""Generate paper-ready SONICOM Q26 final-test tables and figures.

This module is deliberately read-only with respect to datasets and checkpoints.  It
only reformats the already sealed final-test CSV/JSON outputs.  The SVG figures are
dependency-free; PNG previews additionally require Pillow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


METRICS = (
    ("FullSphereERB", "Full-sphere ERB", "Full-sphere ERB"),
    ("Contralateral25ERB", "Contralateral ERB", "Contralateral 25-degree ERB"),
    ("ContralateralHighFrequency", "Contralateral HF", "Contralateral >10 kHz"),
    ("HorizontalILDMAE", "Horizontal ILD", "Horizontal-plane ILD"),
)

METHODS = (
    ("MCA", "MCA", "MCA_Mean_dB"),
    ("MLP v1", "MLP v1", "MLPv1_Mean_dB"),
    ("MLP v2", "MLP v2", "MLPv2_Mean_dB"),
    ("MLP+CNN v3", "v3", "MLPCNNv3_Mean_dB"),
    ("MLP+CNN v3.1", "v3.1", "MLPCNNv31_Mean_dB"),
    ("MLP+CNN v3.2", "v3.2", "MLPCNNv32_Mean_dB"),
)

COLORS = {
    "MCA": "#9CA3AF",
    "MLP v1": "#60A5FA",
    "MLP v2": "#2563EB",
    "v3": "#14B8A6",
    "v3.1": "#F59E0B",
    "v3.2": "#DC2626",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _escape_tex(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace(">", r"$>$")
    )


def _std_map(v32_summary: dict, v31_summary: dict) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    by_metric_v32 = {row["Metric"]: row for row in v32_summary["aggregate"]}
    by_metric_v31 = {row["Metric"]: row for row in v31_summary["aggregate"]}
    std_fields = {
        "MCA": "MCAStd_dB",
        "MLP v1": "MLPv1Std_dB",
        "MLP v2": "MLPv2Std_dB",
        "MLP+CNN v3": "MLPCNNv3Std_dB",
    }
    for metric, _, _ in METRICS:
        for method, field in std_fields.items():
            output[(method, metric)] = float(by_metric_v32[metric][field])
        # The generic MATLAB evaluator names its configurable fifth method v3.1.
        # In the main run it contains v3.2; in the ablation run it contains v3.1.
        output[("MLP+CNN v3.2", metric)] = float(
            by_metric_v32[metric]["MLPCNNv31Std_dB"]
        )
        output[("MLP+CNN v3.1", metric)] = float(
            by_metric_v31[metric]["MLPCNNv31Std_dB"]
        )
    return output


def _write_main_tables(
    output_dir: Path,
    comparison: list[dict[str, str]],
    stds: dict[tuple[str, str], float],
) -> None:
    by_metric = {row["Metric"]: row for row in comparison}
    rows: list[dict[str, object]] = []
    for method, _, mean_field in METHODS:
        row: dict[str, object] = {"Method": method}
        for metric, _, _ in METRICS:
            row[f"{metric}_Mean_dB"] = float(by_metric[metric][mean_field])
            row[f"{metric}_Std_dB"] = stds[(method, metric)]
        rows.append(row)
    fields = ["Method"] + [
        name
        for metric, _, _ in METRICS
        for name in (f"{metric}_Mean_dB", f"{metric}_Std_dB")
    ]
    _write_csv(output_dir / "table_1_main_test_results.csv", rows, fields)

    header = "Method & " + " & ".join(metric[1] for metric in METRICS) + r" \\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{SONICOM Q26 results on 44 held-out test subjects. Values are subject mean $\pm$ standard deviation in dB; lower is better. The best mean is bold.}",
        r"\label{tab:sonicom-final-test}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for metric, _, _ in METRICS:
            value = float(row[f"{metric}_Mean_dB"])
            std = float(row[f"{metric}_Std_dB"])
            cell = f"{value:.3f} $\\pm$ {std:.3f}"
            if row["Method"] == "MLP+CNN v3.2":
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(_escape_tex(str(row["Method"])) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (output_dir / "table_1_main_test_results.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_ablation_tables(
    output_dir: Path, comparison: list[dict[str, str]], stats: dict
) -> None:
    by_metric = {row["Metric"]: row for row in comparison}
    roles = {
        "MLP v2": "Perceptual-loss MLP baseline",
        "MLP+CNN v3": "Frequency CNN",
        "MLP+CNN v3.1": "Horizontal strict-ILD fine-tuning",
        "MLP+CNN v3.2": "Dual global-magnitude / horizontal-ILD sampling",
    }
    fields = {
        "MLP v2": "MLPv2_Mean_dB",
        "MLP+CNN v3": "MLPCNNv3_Mean_dB",
        "MLP+CNN v3.1": "MLPCNNv31_Mean_dB",
        "MLP+CNN v3.2": "MLPCNNv32_Mean_dB",
    }
    rows: list[dict[str, object]] = []
    for method, role in roles.items():
        row: dict[str, object] = {"Method": method, "Role": role}
        for metric, _, _ in METRICS:
            row[f"{metric}_Mean_dB"] = float(by_metric[metric][fields[method]])
        rows.append(row)
    _write_csv(
        output_dir / "table_2_architecture_ablation.csv",
        rows,
        ["Method", "Role"] + [f"{metric}_Mean_dB" for metric, _, _ in METRICS],
    )
    ablation_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Architecture and training-strategy ablation on 44 held-out SONICOM Q26 test subjects. Values are mean errors in dB; lower is better.}",
        r"\label{tab:sonicom-ablation}",
        r"\begin{tabular}{lp{5.2cm}rrrr}",
        r"\toprule",
        "Method & Change & " + " & ".join(metric[1] for metric in METRICS) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for metric, _, _ in METRICS:
            cell = f"{float(row[f'{metric}_Mean_dB']):.3f}"
            if row["Method"] == "MLP+CNN v3.2":
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        ablation_lines.append(
            f"{_escape_tex(str(row['Method']))} & {_escape_tex(str(row['Role']))} & "
            + " & ".join(cells)
            + r" \\"
        )
    ablation_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (output_dir / "table_2_architecture_ablation.tex").write_text(
        "\n".join(ablation_lines), encoding="utf-8"
    )

    stat_rows: list[dict[str, object]] = []
    for comparison_name, key in (("v3.2 - v3", "v32_vs_v3"), ("v3.2 - v3.1", "v32_vs_v31")):
        for item in stats[key]:
            stat_rows.append(
                {
                    "Comparison": comparison_name,
                    "Metric": item["metric"],
                    "MeanReduction_dB": item["mean_reduction_db"],
                    "Bootstrap95CI_Lower_dB": item["bootstrap_95_ci_db"][0],
                    "Bootstrap95CI_Upper_dB": item["bootstrap_95_ci_db"][1],
                    "ImprovedSubjectCount": item["improved_subject_count"],
                    "SubjectCount": 44,
                    "ExactSignTestP_Uncorrected": item["exact_sign_test_two_sided_p"],
                }
            )
    _write_csv(
        output_dir / "table_3_paired_statistics.csv",
        stat_rows,
        list(stat_rows[0]),
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired subject-level comparison of the locked v3.2 model. Positive reductions favor v3.2. Confidence intervals use 100,000 paired bootstrap replicates. Sign-test $p$ values are two-sided and uncorrected.}",
        r"\label{tab:sonicom-paired}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Comparison & Metric & Reduction (dB) & 95\% CI (dB) & Subjects improved \\",
        r"\midrule",
    ]
    for row in stat_rows:
        lower = float(row["Bootstrap95CI_Lower_dB"])
        upper = float(row["Bootstrap95CI_Upper_dB"])
        lines.append(
            f"{_escape_tex(str(row['Comparison']))} & {_escape_tex(str(row['Metric']))} & "
            f"{float(row['MeanReduction_dB']):.4f} & [{lower:.4f}, {upper:.4f}] & "
            f"{row['ImprovedSubjectCount']}/44 \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (output_dir / "table_3_paired_statistics.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:26px;font-weight:700}.label{font-size:17px}.small{font-size:14px}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#D1D5DB;stroke-width:1}.zero{stroke:#111827;stroke-width:2}</style>',
        f'<text class="title" x="{width / 2}" y="38" text-anchor="middle">{title}</text>',
    ]


def _write_svg_and_png(output_dir: Path, stem: str, svg: str, draw_png) -> None:
    (output_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    image = Image.new("RGB", (1800, 1100), "white")
    drawer = ImageDraw.Draw(image)
    font_paths = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    )
    bold_paths = (
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
    )
    normal_path = next((path for path in font_paths if path.exists()), None)
    bold_path = next((path for path in bold_paths if path.exists()), normal_path)
    fonts = {
        "title": ImageFont.truetype(str(bold_path), 42) if bold_path else ImageFont.load_default(),
        "label": ImageFont.truetype(str(normal_path), 27) if normal_path else ImageFont.load_default(),
        "small": ImageFont.truetype(str(normal_path), 23) if normal_path else ImageFont.load_default(),
        "bold": ImageFont.truetype(str(bold_path), 25) if bold_path else ImageFont.load_default(),
    }
    draw_png(image, drawer, fonts)
    image.save(output_dir / f"{stem}.png", dpi=(300, 300), optimize=True)


def _figure_improvement(output_dir: Path, comparison: list[dict[str, str]]) -> None:
    by_metric = {row["Metric"]: row for row in comparison}
    method_fields = (
        ("MLP v1", "MLPv1_Mean_dB"),
        ("MLP v2", "MLPv2_Mean_dB"),
        ("v3", "MLPCNNv3_Mean_dB"),
        ("v3.1", "MLPCNNv31_Mean_dB"),
        ("v3.2", "MLPCNNv32_Mean_dB"),
    )
    values: list[list[float]] = []
    for metric, _, _ in METRICS:
        mca = float(by_metric[metric]["MCA_Mean_dB"])
        values.append(
            [100.0 * (mca - float(by_metric[metric][field])) / mca for _, field in method_fields]
        )
    width, height = 1200, 740
    left, right, top, bottom = 105, 35, 90, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = 25.0
    svg = _svg_start(width, height, "Error reduction relative to MCA on 44 test subjects")
    for tick in range(0, 26, 5):
        y = top + plot_h * (1 - tick / ymax)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        svg.append(f'<text class="small" x="{left-12}" y="{y+5:.1f}" text-anchor="end">{tick}</text>')
    group_w = plot_w / len(METRICS)
    bar_w = group_w * 0.135
    for gi, ((_, short, _), group) in enumerate(zip(METRICS, values)):
        group_left = left + gi * group_w + group_w * 0.1
        for mi, ((method, _), value) in enumerate(zip(method_fields, group)):
            x = group_left + mi * bar_w
            y = top + plot_h * (1 - value / ymax)
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3:.1f}" height="{top+plot_h-y:.1f}" fill="{COLORS[method]}"/>')
        svg.append(f'<text class="label" x="{left+(gi+0.5)*group_w:.1f}" y="{top+plot_h+35}" text-anchor="middle">{short}</text>')
    svg += [
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<text class="label" transform="translate(28 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Error reduction (%)</text>',
    ]
    legend_y = height - 35
    for index, (method, _) in enumerate(method_fields):
        x = 180 + index * 175
        svg.append(f'<rect x="{x}" y="{legend_y-15}" width="22" height="16" fill="{COLORS[method]}"/>')
        svg.append(f'<text class="small" x="{x+30}" y="{legend_y}">{method}</text>')
    svg.append("</svg>")

    def draw_png(image, draw, fonts):
        from PIL import Image as PILImage, ImageDraw as PILImageDraw

        sx, sy = image.width / width, image.height / height
        def xy(values): return tuple(int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(values))
        draw.text((image.width // 2, 35), "Error reduction relative to MCA on 44 test subjects", font=fonts["title"], anchor="ma", fill="#111827")
        for tick in range(0, 26, 5):
            y = (top + plot_h * (1 - tick / ymax)) * sy
            draw.line((left*sx, y, (left+plot_w)*sx, y), fill="#D1D5DB", width=2)
            draw.text((left*sx-18, y), str(tick), font=fonts["small"], anchor="rm", fill="#374151")
        for gi, ((_, short, _), group) in enumerate(zip(METRICS, values)):
            group_left = left + gi * group_w + group_w * 0.1
            for mi, ((method, _), value) in enumerate(zip(method_fields, group)):
                x = group_left + mi * bar_w
                y = top + plot_h * (1 - value / ymax)
                draw.rectangle(xy((x, y, x+bar_w-3, top+plot_h)), fill=COLORS[method])
            draw.text(((left+(gi+0.5)*group_w)*sx, (top+plot_h+38)*sy), short, font=fonts["label"], anchor="ma", fill="#111827")
        draw.line(xy((left, top+plot_h, left+plot_w, top+plot_h)), fill="#374151", width=3)
        draw.line(xy((left, top, left, top+plot_h)), fill="#374151", width=3)
        label_layer = PILImage.new("RGBA", (470, 60), (255, 255, 255, 0))
        label_draw = PILImageDraw.Draw(label_layer)
        label_draw.text(
            (235, 30),
            "Error reduction (%)",
            font=fonts["label"],
            anchor="mm",
            fill="#111827",
        )
        rotated_label = label_layer.rotate(90, expand=True)
        image.paste(rotated_label, (8, 300), rotated_label)
        for index, (method, _) in enumerate(method_fields):
            x = 180 + index * 175
            draw.rectangle(xy((x, legend_y-15, x+22, legend_y+1)), fill=COLORS[method])
            draw.text(((x+30)*sx, legend_y*sy), method, font=fonts["small"], anchor="lm", fill="#111827")
    _write_svg_and_png(output_dir, "figure_1_test_improvement", "\n".join(svg), draw_png)


def _figure_paired(output_dir: Path, stats: dict) -> None:
    panels = (("v3.2 versus v3", stats["v32_vs_v3"]), ("v3.2 versus v3.1", stats["v32_vs_v31"]))
    width, height = 1200, 740
    svg = _svg_start(width, height, "Paired subject bootstrap: error reduction by v3.2")
    # Leave a dedicated label gutter before each panel and a count gutter after
    # it.  This keeps long metric names from being clipped or colliding across
    # the two panels in both the vector and raster renderings.
    panel_boxes = ((210, 105, 330, 500), (790, 105, 330, 500))
    limits = ((-0.005, 0.04), (-0.015, 0.10))
    for (title, items), (x0, y0, pw, ph), (xmin, xmax) in zip(panels, panel_boxes, limits):
        svg.append(f'<text class="label" x="{x0+pw/2}" y="{y0-20}" text-anchor="middle" font-weight="700">{title}</text>')
        zx = x0 + (0-xmin)/(xmax-xmin)*pw
        svg.append(f'<line class="zero" x1="{zx:.1f}" y1="{y0}" x2="{zx:.1f}" y2="{y0+ph}"/>')
        for tick_index in range(6):
            tick = xmin + (xmax-xmin)*tick_index/5
            x = x0 + tick_index*pw/5
            svg.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+ph}"/>')
            svg.append(f'<text class="small" x="{x:.1f}" y="{y0+ph+28}" text-anchor="middle">{tick:.3f}</text>')
        for index, (item, (_, short, _)) in enumerate(zip(items, METRICS)):
            y = y0 + (index+0.65)*ph/4
            mean = float(item["mean_reduction_db"])
            low, high = map(float, item["bootstrap_95_ci_db"])
            mx = x0 + (mean-xmin)/(xmax-xmin)*pw
            lx = x0 + (low-xmin)/(xmax-xmin)*pw
            hx = x0 + (high-xmin)/(xmax-xmin)*pw
            svg.append(f'<line x1="{lx:.1f}" y1="{y:.1f}" x2="{hx:.1f}" y2="{y:.1f}" stroke="#1F2937" stroke-width="4"/>')
            svg.append(f'<circle cx="{mx:.1f}" cy="{y:.1f}" r="8" fill="#DC2626"/>')
            svg.append(f'<text class="small" x="{x0-12}" y="{y+5:.1f}" text-anchor="end">{short}</text>')
            svg.append(f'<text class="small" x="{x0+pw+58}" y="{y+5:.1f}" text-anchor="end">{item["improved_subject_count"]}/44</text>')
        svg.append(f'<text class="small" x="{x0+pw/2}" y="{y0+ph+65}" text-anchor="middle">Mean error reduction (dB; positive favors v3.2)</text>')
    svg.append('<text class="small" x="600" y="710" text-anchor="middle">Dots: paired mean reduction; whiskers: 95% bootstrap CI; labels: subjects improved.</text>')
    svg.append("</svg>")

    def draw_png(image, draw, fonts):
        sx, sy = image.width / width, image.height / height
        draw.text((image.width//2, 35), "Paired subject bootstrap: error reduction by v3.2", font=fonts["title"], anchor="ma", fill="#111827")
        for (title, items), (x0, y0, pw, ph), (xmin, xmax) in zip(panels, panel_boxes, limits):
            draw.text(((x0+pw/2)*sx, (y0-20)*sy), title, font=fonts["bold"], anchor="ma", fill="#111827")
            for tick_index in range(6):
                tick = xmin + (xmax-xmin)*tick_index/5
                x = (x0+tick_index*pw/5)*sx
                draw.line((x,y0*sy,x,(y0+ph)*sy), fill="#D1D5DB", width=2)
                draw.text((x,(y0+ph+30)*sy), f"{tick:.3f}", font=fonts["small"], anchor="ma", fill="#374151")
            zx = (x0+(0-xmin)/(xmax-xmin)*pw)*sx
            draw.line((zx,y0*sy,zx,(y0+ph)*sy), fill="#111827", width=4)
            for index, (item, (_, short, _)) in enumerate(zip(items, METRICS)):
                y = (y0+(index+0.65)*ph/4)*sy
                mean = float(item["mean_reduction_db"])
                low, high = map(float, item["bootstrap_95_ci_db"])
                mx = (x0+(mean-xmin)/(xmax-xmin)*pw)*sx
                lx = (x0+(low-xmin)/(xmax-xmin)*pw)*sx
                hx = (x0+(high-xmin)/(xmax-xmin)*pw)*sx
                draw.line((lx,y,hx,y), fill="#1F2937", width=6)
                draw.ellipse((mx-10,y-10,mx+10,y+10), fill="#DC2626")
                draw.text(((x0-12)*sx,y), short, font=fonts["small"], anchor="rm", fill="#111827")
                draw.text(((x0+pw+58)*sx,y), f"{item['improved_subject_count']}/44", font=fonts["small"], anchor="rm", fill="#374151")
        draw.text((image.width//2, image.height-35), "Dots: paired mean reduction; whiskers: 95% bootstrap CI; labels: subjects improved.", font=fonts["small"], anchor="ms", fill="#374151")
    _write_svg_and_png(output_dir, "figure_2_paired_effects", "\n".join(svg), draw_png)


def _figure_generalization(output_dir: Path, summary: dict) -> None:
    changes = summary["validation_to_test_generalization"]["test_change_relative_to_validation_percent"]
    keys = ("full_sphere_erb", "contralateral_25_degree_erb", "contralateral_high_frequency", "horizontal_ild_mae")
    values = [float(changes[key]) for key in keys]
    width, height = 1200, 700
    left, right, top, bottom = 310, 80, 100, 90
    plot_w, plot_h = width-left-right, height-top-bottom
    xmin, xmax = -4.0, 12.0
    svg = _svg_start(width, height, "Validation-to-test shift for the locked v3.2 model")
    for tick in range(-4, 13, 2):
        x = left+(tick-xmin)/(xmax-xmin)*plot_w
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}"/>')
        svg.append(f'<text class="small" x="{x:.1f}" y="{top+plot_h+30}" text-anchor="middle">{tick}</text>')
    zx = left+(0-xmin)/(xmax-xmin)*plot_w
    svg.append(f'<line class="zero" x1="{zx:.1f}" y1="{top}" x2="{zx:.1f}" y2="{top+plot_h}"/>')
    for index, (value, (_, short, _)) in enumerate(zip(values, METRICS)):
        y = top+(index+0.25)*plot_h/4
        h = plot_h/8
        x1 = left+(min(0,value)-xmin)/(xmax-xmin)*plot_w
        x2 = left+(max(0,value)-xmin)/(xmax-xmin)*plot_w
        color = "#2563EB" if value < 0 else "#F59E0B"
        svg.append(f'<rect x="{x1:.1f}" y="{y:.1f}" width="{x2-x1:.1f}" height="{h:.1f}" fill="{color}"/>')
        svg.append(f'<text class="label" x="{left-18}" y="{y+h*0.72:.1f}" text-anchor="end">{short}</text>')
        anchor = "end" if value < 0 else "start"
        tx = x1-8 if value < 0 else x2+8
        svg.append(f'<text class="label" x="{tx:.1f}" y="{y+h*0.72:.1f}" text-anchor="{anchor}">{value:+.2f}%</text>')
    svg.append(f'<text class="label" x="{left+plot_w/2}" y="{height-20}" text-anchor="middle">Relative error change from validation to test (%)</text>')
    svg.append('<text class="small" x="600" y="72" text-anchor="middle">Negative is lower error on test; positive is higher error on test.</text>')
    svg.append("</svg>")

    def draw_png(image, draw, fonts):
        sx, sy = image.width/width, image.height/height
        draw.text((image.width//2,35), "Validation-to-test shift for the locked v3.2 model", font=fonts["title"], anchor="ma", fill="#111827")
        draw.text((image.width//2,100), "Negative is lower error on test; positive is higher error on test.", font=fonts["small"], anchor="ma", fill="#374151")
        for tick in range(-4,13,2):
            x = (left+(tick-xmin)/(xmax-xmin)*plot_w)*sx
            draw.line((x,top*sy,x,(top+plot_h)*sy), fill="#D1D5DB", width=2)
            draw.text((x,(top+plot_h+30)*sy), str(tick), font=fonts["small"], anchor="ma", fill="#374151")
        zx = (left+(0-xmin)/(xmax-xmin)*plot_w)*sx
        draw.line((zx,top*sy,zx,(top+plot_h)*sy), fill="#111827", width=4)
        for index,(value,(_,short,_)) in enumerate(zip(values,METRICS)):
            y = top+(index+0.25)*plot_h/4
            h = plot_h/8
            x1 = left+(min(0,value)-xmin)/(xmax-xmin)*plot_w
            x2 = left+(max(0,value)-xmin)/(xmax-xmin)*plot_w
            color = "#2563EB" if value < 0 else "#F59E0B"
            draw.rectangle((int(x1*sx),int(y*sy),int(x2*sx),int((y+h)*sy)),fill=color)
            draw.text(((left-18)*sx,(y+h/2)*sy),short,font=fonts["label"],anchor="rm",fill="#111827")
            tx = (x1-8)*sx if value < 0 else (x2+8)*sx
            draw.text((tx,(y+h/2)*sy),f"{value:+.2f}%",font=fonts["bold"],anchor="rm" if value < 0 else "lm",fill="#111827")
    _write_svg_and_png(output_dir, "figure_3_generalization_shift", "\n".join(svg), draw_png)


def generate(project_root: Path, output_dir: Path) -> None:
    source = project_root / "results" / "sonicom_mlp_cnn_q26_v32_final_test"
    strict = project_root / "results" / "sonicom_mlp_cnn_q26_v32_final_test_strict"
    v31_strict = project_root / "results" / "sonicom_mlp_cnn_q26_v31_final_test_strict"
    comparison = _read_csv(source / "comparison.csv")
    stats = _read_json(source / "paired_statistics.json")
    summary = _read_json(source / "summary.json")
    strict_summary = _read_json(strict / "summary.json")
    v31_summary = _read_json(v31_strict / "summary.json")
    if summary.get("status") != "completed_primary_model_unchanged":
        raise ValueError("Final test is not marked completed with the primary model unchanged.")
    if summary.get("split") != "test" or summary.get("subject_count") != 44:
        raise ValueError("Expected the sealed 44-subject test result.")
    output_dir.mkdir(parents=True, exist_ok=True)
    stds = _std_map(strict_summary, v31_summary)
    _write_main_tables(output_dir, comparison, stds)
    _write_ablation_tables(output_dir, comparison, stats)
    _figure_improvement(output_dir, comparison)
    _figure_paired(output_dir, stats)
    _figure_generalization(output_dir, summary)
    manifest = {
        "schema_version": "1.0",
        "created_on": "2026-08-05",
        "source_split": "test",
        "subject_count": 44,
        "primary_model": summary["primary_model"],
        "model_selection_used_test": False,
        "source_files": [
            str((source / "comparison.csv").relative_to(project_root)).replace("\\", "/"),
            str((source / "paired_statistics.json").relative_to(project_root)).replace("\\", "/"),
            str((source / "summary.json").relative_to(project_root)).replace("\\", "/"),
            str((strict / "summary.json").relative_to(project_root)).replace("\\", "/"),
            str((v31_strict / "summary.json").relative_to(project_root)).replace("\\", "/"),
        ],
        "policy": "Reformatting only. No dataset, checkpoint, prediction, or model selection was performed.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sonicom_mlp_cnn_q26_v32_paper"),
    )
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    generate(project_root, output_dir.resolve())
    print(f"Paper assets written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
