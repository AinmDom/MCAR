"""Generate the CSMT English manuscript figures.

The script only reads frozen geometry and aggregate result files. It does not
read subject responses or execute a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 6.4
plt.rcParams["axes.labelsize"] = 7.0
plt.rcParams["xtick.labelsize"] = 6.2
plt.rcParams["ytick.labelsize"] = 6.2
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"


MM = 1.0 / 25.4
INK = "#26343F"
MUTED = "#66727D"
GRID = "#CBD3D9"
BLUE = "#315E8A"
TEAL = "#3B8C88"
AMBER = "#C17B2F"
PURPLE = "#75669A"
GREEN = "#4F8167"


def configure_language(language):
    if language == "zh":
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        })


def rounded_box(ax, x, y, w, h, text, face, edge, fontsize=6.2, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.28,rounding_size=1.2",
        linewidth=0.7,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
        linespacing=1.22,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color=INK, lw=0.9, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def polyarrow(ax, points, color=INK, lw=0.9):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=lw,
        color=color,
        joinstyle="round",
        capstyle="round",
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def export_figure(fig, stem: Path, dpi=600):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def draw_reconstruction_workflow(
    out_dir: Path, qa_dir: Path, require_matplotlib_panel_alignment
):
    """Draw the complete inference chain from sparse HRTFs to dense HRIRs."""
    width_mm, height_mm = 143.5, 82.0
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)
    ax.set_axis_off()

    stages = [
        (1.0, 20.0, "1  Sparse input", BLUE, "#EEF4F8"),
        (23.0, 31.5, "2  MCA baseline", TEAL, "#EEF7F6"),
        (56.5, 27.0, "3  FSC correction", GREEN, "#EEF6F1"),
        (85.5, 33.0, "4  Spectrum assembly", PURPLE, "#F3F0F7"),
        (120.5, 22.0, "5  Output", AMBER, "#FAF3E9"),
    ]
    for x, w, title, color, fill in stages:
        ax.add_patch(FancyBboxPatch(
            (x, 2.0), w, 77.0,
            boxstyle="round,pad=0.25,rounding_size=1.5",
            linewidth=0.8, linestyle=(0, (3, 2)),
            edgecolor=color, facecolor=fill, zorder=0,
        ))
        ax.text(x + 1.2, 6.2, title, ha="left", va="center",
                fontsize=6.2, fontweight="bold", color=color)

    rounded_box(
        ax, 2.5, 14.0, 17.0, 20.0,
        "Sparse binaural\nHRTFs / HRIRs\nat Q observed\ndirections",
        "white", BLUE, fontsize=5.5, weight="bold",
    )
    rounded_box(
        ax, 2.5, 46.0, 17.0, 18.0,
        "Target queries q\n+ sparse observed\nmagnitudes and xyz",
        "white", BLUE, fontsize=5.45,
    )

    rounded_box(
        ax, 24.5, 12.0, 28.5, 11.5,
        "Time alignment\n+ spatial interpolation",
        "white", TEAL, fontsize=5.65, weight="bold",
    )
    rounded_box(
        ax, 24.5, 28.5, 28.5, 11.5,
        "Magnitude correction from\nsmoothed magnitude",
        "white", TEAL, fontsize=5.55,
    )
    rounded_box(
        ax, 24.5, 45.0, 28.5, 12.0,
        "MCA magnitude M_MCA\n+ correction feature C_MCA",
        "white", TEAL, fontsize=5.55, weight="bold",
    )
    rounded_box(
        ax, 24.5, 62.0, 28.5, 11.0,
        "MCA phase + original\noutside-bin complex values",
        "white", PURPLE, fontsize=5.4,
    )
    arrow(ax, (38.75, 23.5), (38.75, 28.5), TEAL, lw=0.75)
    arrow(ax, (38.75, 40.0), (38.75, 45.0), TEAL, lw=0.75)

    rounded_box(
        ax, 58.0, 14.0, 24.0, 18.0,
        "Frozen FSC model\nFiLM-SIREN +\nspectral CNN",
        "white", GREEN, fontsize=5.7, weight="bold",
    )
    rounded_box(
        ax, 58.0, 39.0, 24.0, 13.0,
        "Normalized residual\nr_tilde_H(q, f)",
        "white", GREEN, fontsize=5.6,
    )
    rounded_box(
        ax, 58.0, 59.0, 24.0, 14.0,
        "Denormalize\nr_H^dB = sigma_r\n× r_tilde_H + mu_r",
        "white", GREEN, fontsize=5.35, weight="bold",
    )
    arrow(ax, (70.0, 32.0), (70.0, 39.0), GREEN, lw=0.75)
    arrow(ax, (70.0, 52.0), (70.0, 59.0), GREEN, lw=0.75)

    rounded_box(
        ax, 87.0, 11.0, 30.0, 12.0,
        "Correct 463 selected bins\nM_hat = M_MCA + r_H^dB",
        "white", PURPLE, fontsize=5.55, weight="bold",
    )
    rounded_box(
        ax, 87.0, 29.0, 30.0, 12.0,
        "Selected-bin complex spectrum\n10^(M_hat/20) exp(j phi_MCA)",
        "white", PURPLE, fontsize=5.4,
    )
    rounded_box(
        ax, 87.0, 47.0, 30.0, 20.0,
        "Assemble 513-bin\nsingle-sided spectrum\n463 corrected complex bins\n+ 50 original MCA complex bins",
        "white", PURPLE, fontsize=5.35, weight="bold",
    )
    arrow(ax, (102.0, 23.0), (102.0, 29.0), PURPLE, lw=0.75)
    arrow(ax, (102.0, 41.0), (102.0, 47.0), PURPLE, lw=0.75)

    rounded_box(
        ax, 122.0, 12.0, 19.0, 12.0,
        "Conjugate-symmetric\nextension",
        "white", AMBER, fontsize=5.5,
    )
    rounded_box(
        ax, 122.0, 31.0, 19.0, 11.0,
        "1024-point IFFT",
        "white", AMBER, fontsize=5.7, weight="bold",
    )
    rounded_box(
        ax, 122.0, 49.0, 19.0, 10.0,
        "Retain first\n256 samples",
        "white", AMBER, fontsize=5.5,
    )
    rounded_box(
        ax, 122.0, 66.0, 19.0, 9.0,
        "Dense binaural\nHRIRs / HRTFs",
        "white", AMBER, fontsize=5.55, weight="bold",
    )
    arrow(ax, (131.5, 24.0), (131.5, 31.0), AMBER, lw=0.75)
    arrow(ax, (131.5, 42.0), (131.5, 49.0), AMBER, lw=0.75)
    arrow(ax, (131.5, 59.0), (131.5, 66.0), AMBER, lw=0.75)

    arrow(ax, (19.5, 24.0), (24.5, 17.75), TEAL, lw=0.85)
    polyarrow(ax, [(19.5, 55.0), (21.8, 55.0), (21.8, 76.5),
                   (55.0, 76.5), (55.0, 23.0),
                   (58.0, 23.0)], BLUE, lw=0.8)
    polyarrow(ax, [(53.0, 51.0), (55.5, 51.0), (55.5, 28.0),
                   (58.0, 28.0)], TEAL, lw=0.8)
    polyarrow(ax, [(82.0, 66.0), (84.7, 66.0), (84.7, 17.0),
                   (87.0, 17.0)], GREEN, lw=0.85)
    polyarrow(ax, [(38.75, 73.0), (38.75, 75.0), (84.0, 75.0),
                   (84.0, 35.0),
                   (87.0, 35.0)], PURPLE, lw=0.8)
    polyarrow(ax, [(38.75, 73.0), (38.75, 77.0), (83.2, 77.0),
                   (83.2, 57.0), (87.0, 57.0)], PURPLE, lw=0.8)
    polyarrow(ax, [(117.0, 57.0), (119.5, 57.0), (119.5, 18.0),
                   (122.0, 18.0)], AMBER, lw=0.85)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=[ax], panel_ids=["reconstruction-workflow"],
        json_out=str(
            qa_dir / "fsc_sparse_to_interpolated_hrtf_workflow.alignment.json"
        ),
        tolerance_pt=1.5, strict=True,
    )
    export_figure(fig, out_dir / "fsc_sparse_to_interpolated_hrtf_workflow")
    plt.close(fig)


def draw_architecture(out_dir: Path, qa_dir: Path, require_matplotlib_panel_alignment):
    """Draw the detailed model architecture without the reconstruction chain."""
    width_mm, height_mm = 143.5, 104.0
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)
    ax.set_axis_off()

    columns = [
        (1.0, 18.5, "1  Inputs", BLUE, "#EEF4F8"),
        (21.0, 23.0, "2  FiLM-SIREN", GREEN, "#EEF6F1"),
        (45.5, 23.5, "3  7-channel features", PURPLE, "#F3F0F7"),
        (70.5, 48.0, "4  Binaural spectral CNN", BLUE, "#EEF4F8"),
        (120.0, 22.5, "5  Residual output", AMBER, "#FAF3E9"),
    ]
    for x, w, title, color, fill in columns:
        ax.add_patch(FancyBboxPatch(
            (x, 2.0), w, 95.0,
            boxstyle="round,pad=0.25,rounding_size=1.5",
            linewidth=0.8, linestyle=(0, (3, 2)),
            edgecolor=color, facecolor=fill, zorder=0,
        ))
        ax.text(x + 1.2, 6.0, title, ha="left", va="center",
                fontsize=6.35, fontweight="bold", color=color)

    rounded_box(ax, 2.2, 10.0, 16.1, 22.0,
                "Condition set\nnormalized\nmagnitude\n[B, 2, Q, 463]\nxyz [B, Q, 3]\nmask [B, Q]",
                "white", BLUE, fontsize=5.3)
    rounded_box(ax, 2.2, 40.0, 16.1, 13.0,
                "Query field\ncoordinates\n[B, N, 7]",
                "white", BLUE, fontsize=5.3)
    rounded_box(ax, 2.2, 61.0, 16.1, 27.0,
                "Local context\nnormalized MCA\nnormalized\ncorrection\nnormalized\nlog-frequency\ndirection xyz",
                "white", BLUE, fontsize=5.3)

    ax.text(32.5, 9.0, "Stage D: frozen", ha="center", fontsize=5.55,
            fontweight="bold", color=GREEN)
    base_boxes = [
        (12.0, 9.0, "Condition encoder\nQ-direction set", False),
        (26.0, 7.0, "latent z  [B, 128]", False),
        (38.0, 8.0, "Layer modulator", True),
        (51.0, 16.0, "6-layer FiLM-SIREN\nwidth 256\nFiLM at every layer", True),
        (71.5, 8.5, "Linear 256 → 2", False),
        (84.5, 8.0, "Base residual\n[B, N, 2]", True),
    ]
    for y, h, label, bold in base_boxes:
        rounded_box(ax, 22.5, y, 20.0, h, label, "white", GREEN,
                    fontsize=5.5, weight="bold" if bold else "normal")
    for y0, y1 in ((21.0, 26.0), (33.0, 38.0), (46.0, 51.0)):
        arrow(ax, (32.5, y0), (32.5, y1), GREEN, lw=0.75)
    arrow(ax, (32.5, 67.4), (32.5, 70.4), GREEN, lw=0.75)
    arrow(ax, (32.5, 80.4), (32.5, 83.4), GREEN, lw=0.75)
    polyarrow(ax, [(18.3, 21.0), (20.2, 21.0), (20.2, 16.5), (22.5, 16.5)], BLUE, lw=0.75)
    polyarrow(ax, [(18.3, 46.5), (20.2, 46.5), (20.2, 59.0), (22.5, 59.0)], BLUE, lw=0.75)

    feature_labels = [
        "1  L MCA", "2  L correction", "3  L base residual",
        "4  R MCA", "5  R correction", "6  R base residual",
        "7  log-frequency",
    ]
    for index, label in enumerate(feature_labels):
        y = 10.5 + index * 8.1
        base_channel = index in (2, 5)
        rounded_box(ax, 47.0, y, 20.5, 5.9, label,
                    "#EEF6F1" if base_channel else "white",
                    GREEN if base_channel else PURPLE, fontsize=5.35)
    rounded_box(ax, 47.0, 69.0, 20.5, 9.0,
                "Stack / assemble\n[B, 7, F]", "white", PURPLE,
                fontsize=5.5, weight="bold")
    rounded_box(ax, 47.0, 84.0, 20.5, 7.5,
                "direction xyz  [B, 3]", "white", PURPLE, fontsize=5.35)
    arrow(ax, (57.25, 65.0), (57.25, 69.0), PURPLE, lw=0.75)
    for y in (30.7, 55.0):
        polyarrow(ax, [(42.5, 88.5), (44.5, 88.5), (44.5, y), (47.0, y)], GREEN, lw=0.72)
    polyarrow(ax, [(18.3, 75.0), (19.9, 75.0), (19.9, 94.0),
                   (43.7, 94.0), (43.7, 14.0), (47.0, 14.0)],
              PURPLE, lw=0.7)

    ax.text(94.5, 9.0, "Stage D: trainable", ha="center", fontsize=5.55,
            fontweight="bold", color=BLUE)
    rounded_box(ax, 72.2, 12.0, 20.0, 20.0,
                "CNN stem\nReflectionPad1d(3)\nConv1d 7 → 48\nkernel 7\nGroupNorm(8) + SiLU",
                "white", BLUE, fontsize=5.3, weight="bold")
    rounded_box(ax, 96.0, 12.0, 20.5, 20.0,
                "Direction encoder\nLinear 3 → 64 → 64\nSiLU after\nboth linear layers\ncondition width 64",
                "white", BLUE, fontsize=5.3, weight="bold")
    polyarrow(ax, [(67.5, 73.5), (69.5, 73.5), (69.5, 21.0), (72.2, 21.0)], PURPLE, lw=0.75)
    polyarrow(ax, [(67.5, 87.8), (68.8, 87.8), (68.8, 34.5),
                   (106.25, 34.5), (106.25, 32.0)], PURPLE, lw=0.75)
    ax.text(94.4, 38.5, "FiLM depthwise residual blocks ×4", ha="center",
            fontsize=5.6, fontweight="bold", color=BLUE)
    block_x = [72.5, 83.7, 94.9, 106.1]
    for idx, (x, dilation) in enumerate(zip(block_x, (1, 2, 4, 8))):
        rounded_box(ax, x, 41.0, 9.6, 16.0,
                    f"Block {idx + 1}\ndilation {dilation}\nkernel 7",
                    "white", BLUE, fontsize=5.3, weight="bold")
        if idx < 3:
            arrow(ax, (x + 9.6, 49.0), (block_x[idx + 1], 49.0), BLUE, lw=0.7)
    polyarrow(ax, [(82.2, 32.0), (82.2, 35.0), (72.5, 35.0), (72.5, 41.0)], BLUE, lw=0.75)
    polyarrow(ax, [(106.25, 32.0), (117.2, 32.0), (117.2, 59.0),
                   (77.3, 59.0),
                   (77.3, 57.0)], BLUE, lw=0.65)
    for x in block_x[1:]:
        polyarrow(ax, [(117.2, 59.0), (x + 4.8, 59.0),
                       (x + 4.8, 57.0)], BLUE, lw=0.65)
    ax.text(88.5, 62.5, "depthwise Conv1d + FiLM\npointwise 48 → 96 → 48",
            ha="center", fontsize=5.35, color=MUTED)
    rounded_box(ax, 79.0, 67.0, 31.0, 8.0,
                "Conv1d 48 → 2, kernel 1", "white", BLUE, fontsize=5.5)
    rounded_box(ax, 79.0, 81.0, 31.0, 9.0,
                "CNN increment  C_theta(X, q)\n[B, N, 2]",
                "white", BLUE, fontsize=5.5, weight="bold")
    polyarrow(ax, [(115.7, 49.0), (116.6, 49.0), (116.6, 71.0),
                   (110.0, 71.0)], BLUE, lw=0.75)
    arrow(ax, (94.5, 75.0), (94.5, 81.0), BLUE, lw=0.75)

    rounded_box(ax, 121.5, 13.0, 19.5, 9.0,
                "Base residual\nr_F  [B, N, 2]", "white", GREEN,
                fontsize=5.45, weight="bold")
    rounded_box(ax, 121.5, 34.0, 19.5, 9.0,
                "CNN increment\nC_theta  [B, N, 2]", "white", BLUE,
                fontsize=5.45, weight="bold")
    ax.add_patch(Circle((131.25, 55.0), 2.8, facecolor="white",
                        edgecolor=AMBER, linewidth=0.9))
    ax.text(131.25, 55.0, "+", ha="center", va="center", fontsize=8.2,
            fontweight="bold", color=AMBER)
    polyarrow(ax, [(121.5, 17.5), (120.8, 17.5), (120.8, 55.0),
                   (128.3, 55.0)], GREEN, lw=0.8)
    arrow(ax, (131.25, 43.0), (131.25, 52.0), BLUE, lw=0.8)
    rounded_box(ax, 121.5, 65.0, 19.5, 13.0,
                "Final normalized\nbinaural residual\n[B, N, 2]",
                "white", AMBER, fontsize=5.45, weight="bold")
    arrow(ax, (131.25, 57.8), (131.25, 65.0), AMBER, lw=0.8)
    rounded_box(ax, 121.5, 84.0, 19.5, 8.0,
                "r_H = r_F +\nC_theta(X, q)", "white", AMBER,
                fontsize=5.35, weight="bold")
    arrow(ax, (131.25, 78.0), (131.25, 84.0), AMBER, lw=0.8)
    polyarrow(ax, [(42.5, 90.0), (44.3, 90.0), (44.3, 95.0),
                   (119.2, 95.0), (119.2, 17.5), (121.5, 17.5)], GREEN, lw=0.8)
    polyarrow(ax, [(110.0, 85.5), (117.0, 85.5), (117.0, 38.5),
                   (121.5, 38.5)], BLUE, lw=0.8)

    ax.text(71.75, 101.0,
            "Single prespecified E190 model; the diagram ends at the predicted residual.",
            ha="center", va="center", fontsize=5.7, color=MUTED)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=[ax], panel_ids=["architecture"],
        json_out=str(qa_dir / "fsc_single_model_architecture.alignment.json"),
        tolerance_pt=1.5, strict=True,
    )
    export_figure(fig, out_dir / "fsc_single_model_architecture")
    plt.close(fig)


def read_grid(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "level": row["sparsity_level"],
                    "source": int(row["source_index_zero_based"]),
                    "azimuth": float(row["azimuth_deg"]),
                    "elevation": float(row["elevation_deg"]),
                }
            )
    return rows


def rotate_xyz(x, y, z, yaw=-0.62, pitch=-0.35):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x1 = cy * x + sy * z
    z1 = -sy * x + cy * z
    return x1, cp * y - sp * z1, sp * y + cp * z1


def spherical_xyz(azimuth_deg, elevation_deg):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)


def rotated_curve(kind, value_deg, samples=241):
    values = np.linspace(0.0, 360.0, samples)
    xyz = []
    for value in values:
        if kind == "latitude":
            point = spherical_xyz(value, value_deg)
        else:
            point = spherical_xyz(value_deg, value - 180.0)
        xyz.append(rotate_xyz(*point))
    return np.asarray(xyz)


def draw_sphere_grid(ax):
    for lat in (-60, -30, 0, 30, 60):
        curve = rotated_curve("latitude", lat)
        for i in range(len(curve) - 1):
            front = (curve[i, 2] + curve[i + 1, 2]) / 2 >= 0
            ax.plot(
                curve[i : i + 2, 0],
                curve[i : i + 2, 1],
                color="#AEB9C1",
                linewidth=0.42 if front else 0.34,
                linestyle="-" if front else (0, (1.5, 1.8)),
                alpha=0.70 if front else 0.35,
                zorder=1,
            )
    for lon in range(0, 360, 30):
        curve = rotated_curve("longitude", lon)
        for i in range(len(curve) - 1):
            front = (curve[i, 2] + curve[i + 1, 2]) / 2 >= 0
            ax.plot(
                curve[i : i + 2, 0],
                curve[i : i + 2, 1],
                color="#AEB9C1",
                linewidth=0.42 if front else 0.34,
                linestyle="-" if front else (0, (1.5, 1.8)),
                alpha=0.70 if front else 0.35,
                zorder=1,
            )
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#7E8B95", linewidth=0.75, zorder=2))


def point_category(source, q14, q26):
    if source in q14:
        return "q14"
    if source in q26:
        return "q26"
    return "q50"


def draw_nested_grids(
    out_dir: Path,
    qa_dir: Path,
    require_matplotlib_panel_alignment,
    grid_rows,
    language="en",
):
    configure_language(language)
    levels = ("Q14", "Q26", "Q50")
    by_level = {level: [row for row in grid_rows if row["level"] == level] for level in levels}
    q14 = {row["source"] for row in by_level["Q14"]}
    q26 = {row["source"] for row in by_level["Q26"]}
    q50 = {row["source"] for row in by_level["Q50"]}
    assert q14 < q26 < q50
    assert (len(q14), len(q26), len(q50)) == (14, 26, 50)

    styles = {
        "q14": (BLUE, "o", 22),
        "q26": (TEAL, "s", 21),
        "q50": (AMBER, "^", 24),
    }
    fig, axes = plt.subplots(1, 3, figsize=(143.5 * MM, 58.0 * MM))
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.17, top=0.89, wspace=0.07)
    for label, level, ax in zip(("a", "b", "c"), levels, axes):
        draw_sphere_grid(ax)
        projected = []
        for row in by_level[level]:
            x, y, z = rotate_xyz(*spherical_xyz(row["azimuth"], row["elevation"]))
            projected.append((z, x, y, point_category(row["source"], q14, q26)))
        for z, x, y, category in sorted(projected, key=lambda item: item[0]):
            color, marker, size = styles[category]
            ax.scatter(
                [x],
                [y],
                s=size,
                marker=marker,
                facecolor=color,
                edgecolor="white" if z >= 0 else INK,
                linewidth=0.45,
                alpha=0.96 if z >= 0 else 0.35,
                zorder=5 if z >= 0 else 3,
            )
        ax.set_xlim(-1.10, 1.10)
        ax.set_ylim(-1.10, 1.10)
        ax.set_aspect("equal")
        ax.set_axis_off()
        title = (
            f"{level}（{len(by_level[level])}个方向）"
            if language == "zh"
            else f"{level}  ({len(by_level[level])} directions)"
        )
        ax.set_title(
            title,
            fontsize=7.0,
            fontweight="bold",
            pad=2.0,
            color=INK,
        )
        ax.text(
            -0.02,
            1.05,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=INK,
        )

    legend_labels = (
        ("Q14核心方向", "Q26新增方向", "Q50新增方向")
        if language == "zh"
        else ("Q14 core", "Added for Q26", "Added for Q50")
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.5, markerfacecolor=BLUE, markeredgecolor="white", label=legend_labels[0]),
        Line2D([0], [0], marker="s", linestyle="none", markersize=4.4, markerfacecolor=TEAL, markeredgecolor="white", label=legend_labels[1]),
        Line2D([0], [0], marker="^", linestyle="none", markersize=4.8, markerfacecolor=AMBER, markeredgecolor="white", label=legend_labels[2]),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        fontsize=6.3,
        columnspacing=1.5,
        handletextpad=0.45,
    )
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes),
        panel_ids=["a", "b", "c"],
        row_groups=[["a", "b", "c"]],
        json_out=str(qa_dir / "spherical_grid_triptych_q14_q26_q50.alignment.json"),
        overlay_svg=str(qa_dir / "spherical_grid_triptych_q14_q26_q50.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    suffix = "_zh" if language == "zh" else ""
    export_figure(fig, out_dir / f"spherical_grid_triptych_q14_q26_q50{suffix}")
    plt.close(fig)


def read_erb_summary(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Metric"] == "FullSphereERB":
                rows.append(
                    {
                        "q": int(row["DirectionCount"]),
                        "n": int(row["SubjectCount"]),
                        "mean": float(row["Mean"]),
                        "low": float(row["Bootstrap95Lower"]),
                        "high": float(row["Bootstrap95Upper"]),
                    }
                )
    rows.sort(key=lambda row: row["q"])
    assert [row["q"] for row in rows] == [14, 26, 50]
    assert {row["n"] for row in rows} == {44}
    assert all(row["low"] <= row["mean"] <= row["high"] for row in rows)
    return rows


def draw_erb_trend(
    out_dir: Path,
    qa_dir: Path,
    require_matplotlib_panel_alignment,
    rows,
    language="en",
):
    configure_language(language)
    q = np.array([row["q"] for row in rows], dtype=float)
    mean = np.array([row["mean"] for row in rows])
    low = np.array([row["low"] for row in rows])
    high = np.array([row["high"] for row in rows])
    yerr = np.vstack([mean - low, high - mean])
    assert np.all(np.diff(mean) < 0)

    fig, ax = plt.subplots(figsize=(120.0 * MM, 68.0 * MM))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.21, top=0.94)
    ax.errorbar(
        q,
        mean,
        yerr=yerr,
        color=BLUE,
        linewidth=1.35,
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        ecolor=BLUE,
        elinewidth=1.0,
        capsize=3.0,
        capthick=1.0,
        zorder=3,
    )
    ax.set_xlim(11, 53)
    ax.set_ylim(0.68, 1.145)
    ax.set_xticks(q, ["Q14", "Q26", "Q50"])
    ax.set_yticks(np.arange(0.7, 1.11, 0.1))
    if language == "zh":
        ax.set_xlabel("观测方向数")
        ax.set_ylabel("全域ERB误差 (dB)")
    else:
        ax.set_xlabel("Observed directions")
        ax.set_ylabel("Full-sphere ERB error (dB)")
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75)
    ax.tick_params(direction="out", length=2.5, width=0.65, color=INK)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
    for x, value, upper in zip(q, mean, high):
        ax.text(x, upper + 0.010, f"{value:.3f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        panel_ids=["trend"],
        json_out=str(qa_dir / "matched_density_erb_trend.alignment.json"),
        tolerance_pt=1.5,
        strict=True,
    )
    suffix = "_zh" if language == "zh" else ""
    export_figure(fig, out_dir / f"matched_density_erb_trend{suffix}")
    plt.close(fig)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-scripts", type=Path)
    parser.add_argument(
        "--only",
        choices=("all", "workflow", "architecture", "grids", "trend", "quantitative"),
        default="all",
        help="Render only one figure when revising an accepted submission set.",
    )
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--skip-alignment-qa",
        action="store_true",
        help="Skip the optional external panel-alignment audit during local rendering.",
    )
    args = parser.parse_args()
    if args.language == "zh" and args.only not in ("grids", "trend", "quantitative"):
        parser.error("Chinese rendering is currently supported for grids, trend, or quantitative.")
    if args.skip_alignment_qa:
        def require_matplotlib_panel_alignment(*_args, **_kwargs):
            return None
    else:
        if args.skill_scripts is None:
            parser.error("--skill-scripts is required unless --skip-alignment-qa is used.")
        sys.path.insert(0, str(args.skill_scripts.resolve()))
        from audit_panel_alignment import require_matplotlib_panel_alignment

    source_dir = Path(__file__).resolve().parent
    grand_paper_en = source_dir.parent
    root = source_dir.parents[2]
    out_dir = args.output_dir.resolve() if args.output_dir else grand_paper_en / "figure"
    qa_dir = (
        out_dir.parent / "figure_src" / "qa"
        if args.language == "zh"
        else source_dir / "qa"
    )
    qa_dir.mkdir(parents=True, exist_ok=True)
    grid_csv = root / "configs" / "data" / "sonicom_nested_sparse_grid_q14_q26_q50_v1.csv"
    erb_csv = (
        root
        / "results"
        / "sonicom_fsc_single_member_matched_density_all_metrics_test_v1"
        / "all_metrics_summary_mean_std.csv"
    )
    assert grid_csv.is_file()
    assert erb_csv.is_file()

    grid_rows = read_grid(grid_csv)
    erb_rows = read_erb_summary(erb_csv)
    if args.only in ("all", "workflow"):
        draw_reconstruction_workflow(
            out_dir, qa_dir, require_matplotlib_panel_alignment
        )
    if args.only in ("all", "architecture"):
        draw_architecture(out_dir, qa_dir, require_matplotlib_panel_alignment)
    if args.only in ("all", "grids", "quantitative"):
        draw_nested_grids(
            out_dir,
            qa_dir,
            require_matplotlib_panel_alignment,
            grid_rows,
            language=args.language,
        )
    if args.only in ("all", "trend", "quantitative"):
        draw_erb_trend(
            out_dir,
            qa_dir,
            require_matplotlib_panel_alignment,
            erb_rows,
            language=args.language,
        )

    figure_width_mm = {
        "fsc_sparse_to_interpolated_hrtf_workflow": 143.5,
        "fsc_single_model_architecture": 143.5,
        "spherical_grid_triptych_q14_q26_q50": 143.5,
        "matched_density_erb_trend": 120.0,
    }
    if args.language == "zh":
        figure_width_mm = {
            "spherical_grid_triptych_q14_q26_q50_zh": 143.5,
            "matched_density_erb_trend_zh": 120.0,
        }

    provenance = {
        "schema_version": "1.0",
        "backend": "python/matplotlib",
        "language": args.language,
        "alignment_qa": "skipped" if args.skip_alignment_qa else "enabled",
        "test_subjects_read": 0,
        "figure_width_mm": figure_width_mm,
        "data_sources": {
            str(grid_csv.relative_to(root)).replace("\\", "/"): sha256(grid_csv),
            str(erb_csv.relative_to(root)).replace("\\", "/"): sha256(erb_csv),
        },
        "data_counts": {
            "grid_rows": len(grid_rows),
            "grid_unique_counts": {
                level: len({row["source"] for row in grid_rows if row["level"] == level})
                for level in ("Q14", "Q26", "Q50")
            },
            "erb_summary_rows": len(erb_rows),
            "erb_subjects": 44,
        },
        "statistics": "Mean and 95% subject-bootstrap CI from authoritative aggregate CSV; no significance test added.",
        "formats": ["svg", "pdf", "png@600dpi", "tiff@600dpi_lzw"],
    }
    provenance_dir = out_dir.parent / "figure_src" if args.language == "zh" else source_dir
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_name = "provenance_zh.json" if args.language == "zh" else "provenance.json"
    (provenance_dir / provenance_name).write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
