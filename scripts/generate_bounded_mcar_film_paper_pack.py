"""Generate validation-only paper tables and figures for the Stage-E model."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "results/sonicom_bounded_mcar_film_correction_final_e25_validation"
SINGLE_ROOT = ROOT / "results/sonicom_bounded_mcar_film_correction_d1_e40_validation"
MANIFEST = ROOT / (
    "configs/experiments/"
    "sonicom_bounded_mcar_film_correction_final_e25_ensemble_manifest.json"
)
OUTPUT = ROOT / "results/sonicom_bounded_mcar_film_correction_e25_paper"

METRICS = (
    "FullSphereERB",
    "Contralateral25ERB",
    "ContralateralHighFrequency",
    "HorizontalILDMAE",
)
METRIC_LABELS = {
    "FullSphereERB": "Full-sphere ERB",
    "Contralateral25ERB": "Contralateral-25 ERB",
    "ContralateralHighFrequency": "Contralateral HF",
    "HorizontalILDMAE": "Horizontal ILD",
}
FINAL_METHODS = {
    "HYBRID": "Bounded correction E25 ensemble",
    "PARENT": "Spectral-CNN hybrid E190",
    "FILMENS": "FiLM-SIREN E130 ensemble",
    "MCAR": "MCAR v3.5.1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_map(root: Path) -> dict[tuple[str, str], tuple[float, float]]:
    rows = read_csv(root / "aggregate_metrics.csv")
    return {
        (row["Method"], row["Metric"]): (float(row["Mean_dB"]), float(row["Std_dB"]))
        for row in rows
    }


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def write_main_table(values: dict[tuple[str, str], tuple[float, float]]) -> None:
    rows = []
    method_order = ("HYBRID", "PARENT", "FILMENS", "MCAR")
    for method in method_order:
        row: dict[str, object] = {"Method": FINAL_METHODS[method]}
        for metric in METRICS:
            mean, std = values[(method, metric)]
            row[f"{metric}_Mean_dB"] = mean
            row[f"{metric}_Std_dB"] = std
        rows.append(row)
    fields = ["Method"] + [
        item for metric in METRICS for item in (f"{metric}_Mean_dB", f"{metric}_Std_dB")
    ]
    write_csv(OUTPUT / "table_1_main_validation_results.csv", fields, rows)

    best = {metric: min(values[(method, metric)][0] for method in method_order) for metric in METRICS}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Validation results on 44 SONICOM subjects. Values are subject mean $\pm$ standard deviation in dB; lower is better. Best means are bold.}",
        r"\label{tab:bounded-main-validation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Full-sphere ERB & Contra.-25 ERB & Contra. HF & Horizontal ILD \\",
        r"\midrule",
    ]
    for method in method_order:
        cells = []
        for metric in METRICS:
            mean, std = values[(method, metric)]
            cell = f"{mean:.3f} $\\pm$ {std:.3f}"
            if np.isclose(mean, best[metric]):
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{latex_escape(FINAL_METHODS[method])} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (OUTPUT / "table_1_main_validation_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_paired_table() -> None:
    source = read_csv(FINAL_ROOT / "paired_bootstrap_comparisons.csv")
    rows = []
    for row in source:
        lower = float(row["Bootstrap95Lower_dB"])
        upper = float(row["Bootstrap95Upper_dB"])
        rows.append(
            {
                "Baseline": FINAL_METHODS[row["Baseline"]],
                "Metric": METRIC_LABELS[row["Metric"]],
                "CandidateMinusBaselineMean_dB": float(row["HybridMinusBaselineMean_dB"]),
                "Bootstrap95Lower_dB": lower,
                "Bootstrap95Upper_dB": upper,
                "CandidateWins": int(row["HybridWins"]),
                "Significance": "better" if upper < 0 else "worse" if lower > 0 else "tie",
            }
        )
    fields = list(rows[0])
    write_csv(OUTPUT / "table_2_paired_statistics.csv", fields, rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired validation comparison of the bounded-correction ensemble. Differences are candidate minus baseline; negative values favor the candidate. Intervals use 10,000 subject-paired bootstrap replicates.}",
        r"\label{tab:bounded-paired-validation}",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Baseline & Metric & Difference & 95\% CI & Wins & Result \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['Baseline']))} & {latex_escape(str(row['Metric']))} & "
            f"{row['CandidateMinusBaselineMean_dB']:+.4f} & "
            f"[{row['Bootstrap95Lower_dB']:+.4f}, {row['Bootstrap95Upper_dB']:+.4f}] & "
            f"{row['CandidateWins']}/44 & {row['Significance']} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (OUTPUT / "table_2_paired_statistics.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_development_table(
    final_values: dict[tuple[str, str], tuple[float, float]],
    single_values: dict[tuple[str, str], tuple[float, float]],
) -> None:
    definitions = (
        ("MCAR v3.5.1", final_values, "MCAR", "Frozen convolutional baseline"),
        ("FiLM-SIREN E130 ensemble", final_values, "FILMENS", "Implicit neural representation"),
        ("Spectral-CNN hybrid E190", final_values, "PARENT", "FiLM-SIREN plus spectral CNN"),
        ("Bounded correction, one seed", single_values, "HYBRID", "Frozen MCAR plus 514-parameter gate"),
        ("Bounded correction E25 ensemble", final_values, "HYBRID", "Three formal seeds, equal weights"),
    )
    rows = []
    for name, values, method, note in definitions:
        row: dict[str, object] = {"Method": name, "Role": note}
        for metric in METRICS:
            row[f"{metric}_Mean_dB"] = values[(method, metric)][0]
        rows.append(row)
    fields = ["Method", "Role", *[f"{metric}_Mean_dB" for metric in METRICS]]
    write_csv(OUTPUT / "table_3_development_ablation.csv", fields, rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Development comparison on the same 44 validation subjects. The one-seed bounded model is included only as a development ablation; the formal result uses the fixed three-seed ensemble.}",
        r"\label{tab:bounded-development}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Full-sphere ERB & Contra.-25 ERB & Contra. HF & Horizontal ILD \\",
        r"\midrule",
    ]
    for row in rows:
        values = [float(row[f"{metric}_Mean_dB"]) for metric in METRICS]
        lines.append(
            f"{latex_escape(str(row['Method']))} & "
            + " & ".join(f"{value:.3f}" for value in values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (OUTPUT / "table_3_development_ablation.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    final_values = aggregate_map(FINAL_ROOT)
    single_values = aggregate_map(SINGLE_ROOT)
    write_main_table(final_values)
    write_paired_table()
    write_development_table(final_values, single_values)
    manifest = {
        "schema_version": "1.0",
        "created_on": "2026-08-29",
        "scope": "Validation-only paper artifact generation; no model selection or inference.",
        "subject_count": 44,
        "primary_model": "Bounded MCAR plus FiLM-SIREN correction E25 three-seed ensemble",
        "model_manifest_identity": json.loads(MANIFEST.read_text(encoding="utf-8"))[
            "identity_sha256"
        ],
        "source_files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for path in (
                FINAL_ROOT / "aggregate_metrics.csv",
                FINAL_ROOT / "paired_bootstrap_comparisons.csv",
                FINAL_ROOT / "summary.json",
                SINGLE_ROOT / "aggregate_metrics.csv",
                MANIFEST,
                ROOT / "scripts/generate_bounded_mcar_film_paper_pack.py",
                ROOT / "scripts/generate_bounded_mcar_film_paper_figures.m",
            )
        ],
        "policy": (
            "Reformatting and visualization only. No dataset, checkpoint, prediction, "
            "test subject, tuning, or model-selection operation was performed."
        ),
        "test_subject_count_read": 0,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
