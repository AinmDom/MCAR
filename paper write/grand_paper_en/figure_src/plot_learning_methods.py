import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CSV = SCRIPT_DIR / "data" / "learning_methods_plot.csv"
FIGURE_DIR = SCRIPT_DIR.parent / "figure"

# ============================================================
# Plot configuration
# ============================================================
METHOD_ORDER = ["FSC", "RANF", "FSP-AE"]

METRICS_EN = [
    ("FullSphereERB", "Full-sphere ERB", "dB"),
    ("Contralateral25ERB", "Contralateral 25° ERB", "dB"),
    ("ERBBandILDMean", "ERB-band ILD", "dB"),
    ("ITDWeightedMAE", "ITD-weighted MAE", "μs"),
    ("FullSphereLSD", "Full-sphere LSD", "dB"),
]

METRICS_ZH = [
    ("FullSphereERB", "全域ERB", "dB"),
    ("Contralateral25ERB", "对侧25° ERB", "dB"),
    ("ERBBandILDMean", "频带ILD", "dB"),
    ("ITDWeightedMAE", "ITD加权MAE", "μs"),
    ("FullSphereLSD", "全域LSD", "dB"),
]


def configure_fonts(language):
    if language == "zh":
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        })


def render(language="en", output_dir=None):
    if language not in {"en", "zh"}:
        raise ValueError("language must be 'en' or 'zh'")
    configure_fonts(language)
    metrics = METRICS_ZH if language == "zh" else METRICS_EN
    figure_dir = Path(output_dir) if output_dir is not None else FIGURE_DIR
    suffix = "_zh" if language == "zh" else ""
    out_pdf = figure_dir / f"learning_methods_comparison{suffix}.pdf"
    out_png = figure_dir / f"learning_methods_comparison{suffix}.png"

    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{DATA_CSV}\n\n"
            "Expected location:\n"
            "grand_paper_en/figure_src/data/learning_methods_plot.csv"
        )

    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CSV)

    required_columns = {"Method", "Metric", "Mean", "SD"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[df["Method"].isin(METHOD_ORDER)].copy()
    df["Method"] = pd.Categorical(
        df["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(12.8, 3.15),
        constrained_layout=True,
    )

    for ax, (metric_key, title, unit) in zip(axes, metrics):
        sub = df[df["Metric"] == metric_key].copy()
        sub = sub.sort_values("Method")

        if len(sub) != len(METHOD_ORDER):
            raise ValueError(
                f"Metric '{metric_key}' should contain exactly "
                f"{len(METHOD_ORDER)} rows ({METHOD_ORDER}), "
                f"but found {len(sub)}."
            )

        x = list(range(len(METHOD_ORDER)))
        means = sub["Mean"].to_numpy()
        sds = sub["SD"].to_numpy()

        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="o",
            capsize=3,
            markersize=5.5,
            linewidth=1.2,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)

        # Give more space on both left and right sides.
        ax.set_xlim(-0.45, len(METHOD_ORDER) - 1 + 0.45)

        # Add vertical headroom for boxed labels.
        ymin = float((means - sds).min())
        ymax = float((means + sds).max())
        span = max(ymax - ymin, 1e-9)

        ax.set_ylim(
            ymin - 0.10 * span,
            ymax + 0.22 * span,
        )

        # Put mean values inside a small box above the upper error bar.
        label_offset = 0.05 * span
        for xi, mean, sd in zip(x, means, sds):
            ax.text(
                xi,
                mean + sd + label_offset,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.6,
                ),
            )

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved PNG: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    render(args.language, args.output_dir)


if __name__ == "__main__":
    main()
