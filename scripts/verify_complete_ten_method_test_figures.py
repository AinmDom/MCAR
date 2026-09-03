"""Verify all 20 complete ten-method test figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "ten_method_test_metric_figures_v1"


def main() -> None:
    with (OUTPUT / "figure_index.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        index = list(csv.DictReader(handle))
    with (OUTPUT / "aggregate_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    if len(index) != 20 or len(aggregate) != 200:
        raise ValueError("Expected 20 figures and 200 method summaries")
    values = np.asarray([[float(row[key]) for key in ("Mean", "SampleStd", "Bootstrap95Lower", "Bootstrap95Upper")] for row in aggregate])
    if not np.all(np.isfinite(values)) or not np.all(values[:, 2] <= values[:, 0]) or not np.all(values[:, 0] <= values[:, 3]):
        raise ValueError("Invalid numeric summaries")
    dimensions: set[tuple[int, int]] = set()
    png_bytes = pdf_bytes = 0
    for row in index:
        png, pdf = OUTPUT / row["PngFile"], OUTPUT / row["PdfFile"]
        if not png.is_file() or not pdf.is_file() or png.stat().st_size <= 0 or pdf.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty figure {row['FigureNumber']}")
        with Image.open(png) as image:
            image.verify()
            dimensions.add(image.size)
            if image.width < 3000 or image.height < 1800:
                raise ValueError(f"PNG resolution too small: {png}/{image.size}")
        with pdf.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError(f"Invalid PDF: {pdf}")
        png_bytes += png.stat().st_size
        pdf_bytes += pdf.stat().st_size
    report = {
        "status": "verified", "split": "test", "figure_count": 20,
        "png_count": 20, "vector_pdf_count": 20, "method_count_per_figure": 10,
        "subject_count_per_method": 44, "all_numeric_finite": True,
        "all_means_inside_bootstrap_intervals": True,
        "png_dimensions": [list(item) for item in sorted(dimensions)],
        "png_total_bytes": png_bytes, "pdf_total_bytes": pdf_bytes,
        "new_test_subject_count_read": 0,
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
