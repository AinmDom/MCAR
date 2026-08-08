"""Render intuitive FSP-AE horizontal-comparison figures from locked CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 1180
SCALE = 1
WHITE = "#FFFFFF"
INK = "#111827"
MUTED = "#475569"
GRID = "#CBD5E1"
PANEL = "#F8FAFC"
MCA = "#64748B"
MCAR = "#2563EB"
FSP = "#F59E0B"
TEAL = "#14B8A6"
RED = "#DC2626"
OTHER = "#CBD5E1"

METHOD_ORDER = [
    "SHOnly",
    "SUpDEqSH",
    "SUpDEqNN",
    "SUpDEqBary",
    "MCA",
    "MCARv32",
    "FSPAE",
]
METHOD_LABELS = {
    "SHOnly": "SH only",
    "SUpDEqSH": "SUpDEq + SH",
    "SUpDEqNN": "SUpDEq + NN",
    "SUpDEqBary": "SUpDEq + Bary",
    "MCA": "MCA",
    "MCARv32": "MCAR v3.2",
    "FSPAE": "FSP-AE-Q26",
}
METHOD_COLORS = {
    "SHOnly": OTHER,
    "SUpDEqSH": "#B6C2D1",
    "SUpDEqNN": "#9AA8BA",
    "SUpDEqBary": "#7F8DA1",
    "MCA": MCA,
    "MCARv32": MCAR,
    "FSPAE": FSP,
}
METRICS = [
    ("FullSphereERB", "Full-sphere ERB"),
    ("Contralateral25ERB", "Contralateral 25° ERB"),
    ("ContralateralHighFrequency", "Contralateral high frequency"),
    ("HorizontalILDMAE", "Horizontal strict ILD"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    return ImageFont.truetype(str(path), size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, face: Any) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=face)
    return box[2] - box[0], box[3] - box[1]


def centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    face: Any,
    fill: str = INK,
) -> None:
    width, height = text_size(draw, text, face)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=face, fill=fill)


def right_aligned(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    face: Any,
    fill: str = INK,
) -> None:
    width, height = text_size(draw, text, face)
    draw.text((xy[0] - width, xy[1] - height / 2), text, font=face, fill=fill)


def load_aggregate(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["Method"], row["Metric"])] = {
                "mean": float(row["Mean_dB"]),
                "std": float(row["Std_dB"]),
            }
    expected = len(METHOD_ORDER) * len(METRICS)
    if len(values) != expected:
        raise ValueError(f"expected {expected} aggregate cells, found {len(values)}")
    return values


def load_subjects(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 44:
        raise ValueError(f"expected 44 validation subjects, found {len(rows)}")
    return rows


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    return image, ImageDraw.Draw(image)


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    centered(draw, (WIDTH / 2, 62), title, font(40, True))
    centered(draw, (WIDTH / 2, 112), subtitle, font(21), MUTED)


def footer(draw: ImageDraw.ImageDraw) -> None:
    centered(
        draw,
        (WIDTH / 2, HEIGHT - 28),
        "Fixed 44-subject SONICOM validation; 767 interpolation-only directions; test not used.",
        font(17),
        MUTED,
    )


def nice_upper(value: float) -> float:
    raw = value * 1.14
    magnitude = 10 ** math.floor(math.log10(raw))
    scaled = raw / magnitude
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if scaled <= step:
            return step * magnitude
    return 12.0 * magnitude


def vertical_text(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    face: Any,
    fill: str = MUTED,
) -> None:
    temporary = Image.new("RGBA", (260, 50), (255, 255, 255, 0))
    temporary_draw = ImageDraw.Draw(temporary)
    centered(temporary_draw, (130, 25), text, face, fill)
    rotated = temporary.rotate(90, expand=True)
    image.paste(
        rotated,
        (center[0] - rotated.width // 2, center[1] - rotated.height // 2),
        rotated,
    )


def draw_overview_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    metric: str,
    aggregate: dict[tuple[str, str], dict[str, float]],
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=16, fill=PANEL, outline=GRID, width=2)
    centered(draw, ((left + right) / 2, top + 34), title, font(23, True))
    plot_left, plot_right = left + 190, right - 42
    plot_top, plot_bottom = top + 76, bottom - 38
    maximum = max(
        aggregate[(method, metric)]["mean"] + aggregate[(method, metric)]["std"]
        for method in METHOD_ORDER
    )
    x_max = nice_upper(maximum)
    for tick_index in range(5):
        value = x_max * tick_index / 4
        x = plot_left + (plot_right - plot_left) * value / x_max
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
        centered(draw, (x, plot_bottom + 20), f"{value:g}", font(15), MUTED)
    row_height = (plot_bottom - plot_top) / len(METHOD_ORDER)
    for index, method in enumerate(METHOD_ORDER):
        center_y = plot_top + row_height * (index + 0.5)
        entry = aggregate[(method, metric)]
        bar_right = plot_left + (plot_right - plot_left) * entry["mean"] / x_max
        error_left = plot_left + (plot_right - plot_left) * max(
            entry["mean"] - entry["std"], 0.0
        ) / x_max
        error_right = plot_left + (plot_right - plot_left) * (
            entry["mean"] + entry["std"]
        ) / x_max
        right_aligned(
            draw,
            (plot_left - 12, center_y),
            METHOD_LABELS[method],
            font(16, method in {"MCARv32", "FSPAE"}),
            INK,
        )
        draw.rounded_rectangle(
            (plot_left, center_y - 12, bar_right, center_y + 12),
            radius=6,
            fill=METHOD_COLORS[method],
        )
        draw.line((error_left, center_y, error_right, center_y), fill=INK, width=3)
        draw.line((error_left, center_y - 5, error_left, center_y + 5), fill=INK, width=2)
        draw.line((error_right, center_y - 5, error_right, center_y + 5), fill=INK, width=2)
        label_x = min(error_right + 10, plot_right - 45)
        draw.text(
            (label_x, center_y - 9),
            f"{entry['mean']:.3f}",
            font=font(14, method in {"MCARv32", "FSPAE"}),
            fill=INK,
        )


def figure_overview(
    output: Path, aggregate: dict[tuple[str, str], dict[str, float]]
) -> None:
    image, draw = new_canvas()
    header(
        draw,
        "SONICOM Q26 horizontal comparison",
        "Strict interpolation error: mean ± subject SD (dB); lower is better",
    )
    margin_x, gap_x = 55, 28
    panel_width = (WIDTH - 2 * margin_x - gap_x) // 2
    panel_height = 445
    for index, (metric, title) in enumerate(METRICS):
        column, row = index % 2, index // 2
        left = margin_x + column * (panel_width + gap_x)
        top = 150 + row * (panel_height + 24)
        draw_overview_panel(
            draw,
            (left, top, left + panel_width, top + panel_height),
            title,
            metric,
            aggregate,
        )
    footer(draw)
    image.save(output, format="PNG", optimize=True)


def draw_scatter_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    metric: str,
    title: str,
    rows: list[dict[str, str]],
) -> int:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=16, fill=PANEL, outline=GRID, width=2)
    centered(draw, ((left + right) / 2, top + 34), title, font(23, True))
    plot_left, plot_right = left + 82, right - 36
    plot_top, plot_bottom = top + 78, bottom - 70
    x_values = [float(row[f"MCARv32_{metric}_dB"]) for row in rows]
    y_values = [float(row[f"FSPAE_{metric}_dB"]) for row in rows]
    low = min(x_values + y_values)
    high = max(x_values + y_values)
    padding = max((high - low) * 0.12, 0.02)
    low, high = max(0.0, low - padding), high + padding

    def px(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    for tick_index in range(5):
        value = low + (high - low) * tick_index / 4
        x, y = px(value), py(value)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        centered(draw, (x, plot_bottom + 19), f"{value:.2f}", font(14), MUTED)
        right_aligned(draw, (plot_left - 8, y), f"{value:.2f}", font(14), MUTED)
    draw.line((px(low), py(low), px(high), py(high)), fill=INK, width=3)
    wins = 0
    for x_value, y_value in zip(x_values, y_values):
        wins += int(y_value < x_value)
        x, y = px(x_value), py(y_value)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=FSP, outline=WHITE, width=1)
    draw.text((plot_left + 8, plot_top + 8), "FSP better", font=font(15, True), fill=TEAL)
    right_aligned(
        draw,
        (plot_right - 8, plot_bottom - 14),
        "MCAR better",
        font(15, True),
        RED,
    )
    centered(draw, ((plot_left + plot_right) / 2, bottom - 28), "MCAR v3.2 error (dB)", font(16), MUTED)
    vertical_text(
        image,
        (left + 33, (plot_top + plot_bottom) // 2),
        "FSP-AE error (dB)",
        font(15),
    )
    right_aligned(draw, (right - 20, top + 65), f"FSP wins {wins}/44", font(18, True), INK)
    return wins


def figure_scatter(output: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    image, draw = new_canvas()
    header(
        draw,
        "FSP-AE versus MCAR v3.2: paired subjects",
        "Each dot is one validation subject; below the diagonal means lower FSP-AE error",
    )
    margin_x, gap_x = 58, 30
    panel_width = (WIDTH - 2 * margin_x - gap_x) // 2
    panel_height = 445
    wins: dict[str, int] = {}
    for index, (metric, title) in enumerate(METRICS):
        column, row = index % 2, index // 2
        left = margin_x + column * (panel_width + gap_x)
        top = 150 + row * (panel_height + 24)
        wins[metric] = draw_scatter_panel(
            image,
            draw,
            (left, top, left + panel_width, top + panel_height),
            metric,
            title,
            rows,
        )
    footer(draw)
    image.save(output, format="PNG", optimize=True)
    return wins


def improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def figure_tradeoff(
    output: Path,
    aggregate: dict[tuple[str, str], dict[str, float]],
    wins: dict[str, int],
) -> dict[str, dict[str, float]]:
    image, draw = new_canvas()
    header(
        draw,
        "Where FSP-AE helps — and where it does not",
        "Error reduction by FSP-AE; positive is better, negative is worse",
    )
    plot_left, plot_right = 385, 1515
    plot_top, plot_bottom = 220, 930
    minimum, maximum = -40.0, 40.0

    def px(value: float) -> float:
        return plot_left + (value - minimum) / (maximum - minimum) * (plot_right - plot_left)

    for value in range(-40, 41, 10):
        x = px(value)
        draw.line((x, plot_top, x, plot_bottom), fill=INK if value == 0 else GRID, width=4 if value == 0 else 1)
        centered(draw, (x, plot_bottom + 28), f"{value:+d}%" if value else "0%", font(17), MUTED)
    comparisons: dict[str, dict[str, float]] = {}
    row_height = (plot_bottom - plot_top) / len(METRICS)
    for index, (metric, title) in enumerate(METRICS):
        center_y = plot_top + row_height * (index + 0.5)
        fsp = aggregate[("FSPAE", metric)]["mean"]
        vs_mca = improvement(aggregate[("MCA", metric)]["mean"], fsp)
        vs_mcar = improvement(aggregate[("MCARv32", metric)]["mean"], fsp)
        comparisons[metric] = {"vs_mca_percent": vs_mca, "vs_mcar_percent": vs_mcar}
        right_aligned(draw, (plot_left - 32, center_y), title, font(22, True), INK)
        for offset, value, color, label in (
            (-24, vs_mca, TEAL, "vs MCA"),
            (24, vs_mcar, FSP, "vs MCAR"),
        ):
            y = center_y + offset
            x0, x1 = px(0), px(value)
            draw.rounded_rectangle(
                (min(x0, x1), y - 15, max(x0, x1), y + 15),
                radius=7,
                fill=color,
            )
            label_x = x1 + 10 if value >= 0 else x1 - 10
            if value >= 0:
                draw.text((label_x, y - 10), f"{value:+.1f}%", font=font(17, True), fill=INK)
            else:
                right_aligned(draw, (label_x, y), f"{value:+.1f}%", font(17, True), INK)
        draw.text((plot_right + 25, center_y - 11), f"wins vs MCAR: {wins[metric]}/44", font=font(17), fill=MUTED)
    legend_y = 1010
    draw.rounded_rectangle((590, legend_y - 13, 626, legend_y + 13), radius=5, fill=TEAL)
    draw.text((640, legend_y - 11), "FSP-AE versus MCA", font=font(19), fill=INK)
    draw.rounded_rectangle((950, legend_y - 13, 986, legend_y + 13), radius=5, fill=FSP)
    draw.text((1000, legend_y - 11), "FSP-AE versus MCAR v3.2", font=font(19), fill=INK)
    footer(draw)
    image.save(output, format="PNG", optimize=True)
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/sonicom_fsp_ae_q26_formal_validation"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = load_aggregate(args.input_dir / "aggregate_metrics.csv")
    subjects = load_subjects(args.input_dir / "per_subject_metrics.csv")
    figure_overview(output_dir / "figure_1_method_overview.png", aggregate)
    wins = figure_scatter(
        output_dir / "figure_2_fsp_vs_mcar_subjects.png", subjects
    )
    comparisons = figure_tradeoff(
        output_dir / "figure_3_fsp_tradeoff.png", aggregate, wins
    )
    manifest = {
        "status": "completed",
        "source_split": "validation",
        "subject_count": len(subjects),
        "test_subject_count_read": 0,
        "figures": [
            "figure_1_method_overview.png",
            "figure_2_fsp_vs_mcar_subjects.png",
            "figure_3_fsp_tradeoff.png",
        ],
        "fsp_wins_vs_mcar": wins,
        "fsp_improvement_percent": comparisons,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(manifest)


if __name__ == "__main__":
    main()
