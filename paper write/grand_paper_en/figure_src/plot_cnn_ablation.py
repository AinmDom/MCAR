import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CSV = SCRIPT_DIR / "data" / "cnn_ablation_paired_bootstrap.csv"
FIGURE_DIR = SCRIPT_DIR.parent / "figure"

# ============================================================
# Plot configuration
# ============================================================
REPEAT_ORDER = ["Repeat 1", "Repeat 2", "Repeat 3"]

METRICS_EN = [
    ("FullSphereERB", "Full-sphere ERB"),
    ("Contralateral25ERB", "Contralateral 25° ERB"),
    ("ERBBandILDMean", "ERB-band ILD"),
    ("ITDWeightedMAE", "ITD-weighted MAE"),
    ("FullSphereLSD", "Full-sphere LSD"),
    ("HFFirstDifference", "HF first difference"),
    ("HFSecondDifference", "HF second difference"),
    ("MultiscaleNotchDepth", "Multiscale notch depth"),
]

METRICS_ZH = [
    ("FullSphereERB", "全域ERB"),
    ("Contralateral25ERB", "对侧25° ERB"),
    ("ERBBandILDMean", "频带ILD"),
    ("ITDWeightedMAE", "ITD加权MAE"),
    ("FullSphereLSD", "全域LSD"),
    ("HFFirstDifference", "高频一阶差分"),
    ("HFSecondDifference", "高频二阶差分"),
    ("MultiscaleNotchDepth", "多尺度凹口深度"),
]

REPEAT_OFFSETS = {
    "Repeat 1": -0.18,
    "Repeat 2": 0.00,
    "Repeat 3": 0.18,
}


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
    repeat_labels = (
        {"Repeat 1": "重复1", "Repeat 2": "重复2", "Repeat 3": "重复3"}
        if language == "zh"
        else {repeat: repeat for repeat in REPEAT_ORDER}
    )
    figure_dir = Path(output_dir) if output_dir is not None else FIGURE_DIR
    suffix = "_zh" if language == "zh" else ""
    out_pdf = figure_dir / f"cnn_ablation_forest{suffix}.pdf"
    out_png = figure_dir / f"cnn_ablation_forest{suffix}.png"

    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{DATA_CSV}\n\n"
            "Expected location:\n"
            "grand_paper_en/figure_src/data/cnn_ablation_paired_bootstrap.csv"
        )

    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CSV)

    required_columns = {"Metric", "Repeat", "Delta", "CI_Low", "CI_High"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[df["Repeat"].isin(REPEAT_ORDER)].copy()

    metric_keys = [key for key, _ in metrics]
    metric_labels = {key: label for key, label in metrics}

    df["Metric"] = pd.Categorical(
        df["Metric"],
        categories=metric_keys,
        ordered=True,
    )
    df["Repeat"] = pd.Categorical(
        df["Repeat"],
        categories=REPEAT_ORDER,
        ordered=True,
    )
    df = df.sort_values(["Metric", "Repeat"])

    expected_rows = len(metric_keys) * len(REPEAT_ORDER)
    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows "
            f"({len(metric_keys)} metrics × {len(REPEAT_ORDER)} repeats), "
            f"but found {len(df)}."
        )

    # Reverse y positions so the first metric is shown at the top.
    base_y = np.arange(len(metric_keys))[::-1]
    y_by_metric = dict(zip(metric_keys, base_y))

    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)

    for repeat in REPEAT_ORDER:
        sub = df[df["Repeat"] == repeat]

        first_point = True
        for _, row in sub.iterrows():
            metric = str(row["Metric"])
            delta = float(row["Delta"])
            ci_low = float(row["CI_Low"])
            ci_high = float(row["CI_High"])

            if not (ci_low <= delta <= ci_high):
                raise ValueError(
                    f"Inconsistent CI for {metric}, {repeat}: "
                    f"CI_Low={ci_low}, Delta={delta}, CI_High={ci_high}"
                )

            y = y_by_metric[metric] + REPEAT_OFFSETS[repeat]

            # Matplotlib expects asymmetric x errors as [[left], [right]].
            xerr = np.array([
                [delta - ci_low],
                [ci_high - delta],
            ])

            ax.errorbar(
                delta,
                y,
                xerr=xerr,
                fmt="o",
                capsize=3,
                markersize=5,
                linewidth=1.2,
                label=repeat_labels[repeat] if first_point else None,
            )
            first_point = False

    # Zero-effect reference.
    ax.axvline(0.0, linestyle="--", linewidth=1.0)

    ax.set_yticks(base_y)
    ax.set_yticklabels([metric_labels[key] for key in metric_keys])

    if language == "zh":
        ax.set_xlabel(
            r"配对差 $\Delta = E_{\mathrm{E190}} - E_{\mathrm{E130}}$"
            "\n（负值表示CNN补偿后误差降低）"
        )
        legend_title = "独立重复"
    else:
        ax.set_xlabel(
            r"Paired difference $\Delta = E_{\mathrm{E190}} - E_{\mathrm{E130}}$"
            "\n(negative values indicate lower error after CNN refinement)"
        )
        legend_title = "Independent repeat"

    ax.grid(axis="x", alpha=0.25)
    ax.legend(title=legend_title, frameon=False, ncol=3, loc="lower left")

    # Leave a little room to the right of zero so the reference line is visible.
    xmin = min(df["CI_Low"].min(), -0.01)
    xmax = max(df["CI_High"].max(), 0.01)
    pad = 0.06 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)

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
