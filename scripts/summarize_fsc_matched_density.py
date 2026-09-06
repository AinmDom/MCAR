"""Extract the FSC diagonal Q14/Q26/Q50 density table from strict evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/sonicom_fsc_matched_density_direction_sensitivity_v1"
OUT = ROOT / "results/sonicom_fsc_matched_density_v1"
OUT.mkdir(parents=True, exist_ok=False)

rows = []
with (SOURCE / "metric_long.csv").open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        if row["Method"] == "HYBRID":
            row["DirectionCount"] = int(row["DirectionCount"])
            row["SubjectID"] = int(row["SubjectID"])
            row["Value_dB"] = float(row["Value_dB"])
            rows.append(row)
counts = (14, 26, 50)
metrics = ("FullSphereERB", "Contralateral25ERB", "ContralateralHighFrequency", "HorizontalILDMAE")
if len(rows) != 44 * 3 * 4 or not all(np.isfinite(r["Value_dB"]) for r in rows):
    raise RuntimeError("FSC diagonal rows are incomplete or non-finite")
with (OUT / "subject_level.csv").open("w", encoding="utf-8", newline="") as stream:
    fields = ["SubjectLabel", "SubjectID", "DirectionCount", "Method", "MethodLabel", "Metric", "MetricLabel", "Value_dB"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)

summary = []
for metric in metrics:
    for q in counts:
        values = np.asarray([r["Value_dB"] for r in rows if r["Metric"] == metric and r["DirectionCount"] == q], dtype=float)
        if values.size != 44:
            raise RuntimeError(f"incomplete {metric} Q{q}")
        summary.append({"Method": "HYBRID", "MethodLabel": "FSC / Hybrid E190", "DirectionCount": q, "Metric": metric, "SubjectCount": 44, "Mean_dB": float(values.mean()), "SD_dB": float(values.std(ddof=1)), "Min_dB": float(values.min()), "Max_dB": float(values.max())})
with (OUT / "summary_mean_std.csv").open("w", encoding="utf-8", newline="") as stream:
    fields = list(summary[0]); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(summary)

paper = []
for metric in metrics:
    item = {"Metric": metric}
    for q in counts:
        row = next(r for r in summary if r["Metric"] == metric and r["DirectionCount"] == q)
        item[f"Q{q}_Mean_dB"] = row["Mean_dB"]; item[f"Q{q}_SD_dB"] = row["SD_dB"]
    paper.append(item)
with (OUT / "paper_observation_density_table.csv").open("w", encoding="utf-8", newline="") as stream:
    fields = list(paper[0]); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(paper)

metadata = {"status": "completed", "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "method": "HYBRID", "diagonal_counts": list(counts), "subject_count_per_cell": 44, "metrics": list(metrics), "mask": "common Q50-excluded 743-direction mask", "test_subject_count_read": 0, "all_finite": True}
(OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
(OUT / "README.md").write_text("# FSC matched-density summary\n\nDiagonal validation only: FSC-Q14@Q14, FSC-Q26@Q26 (existing frozen validation), and FSC-Q50@Q50. Metrics and the common 743-direction mask are inherited from the strict ten-method evaluator.\n\nFiles: `subject_level.csv`, `summary_mean_std.csv`, and `paper_observation_density_table.csv`. Test subjects were not read.\n", encoding="utf-8")
print(OUT.relative_to(ROOT))
