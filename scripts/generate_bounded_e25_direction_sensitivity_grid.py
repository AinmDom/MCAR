"""Generate the nested Q14/Q26/Q50 grid for Bounded-E25 sensitivity.

Q14 and Q26 are inherited without modification from the two existing frozen
SONICOM grids.  Q50 extends Q26 by twelve left/right reflection pairs chosen
greedily with the same maximin-first, SH-logdet-second ordering used by the
original Q26 geometry generator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mcar.data_tools.prepare_sonicom_configs import (
    build_lateral_reflection_orbits,
    minimum_pairwise_angle_deg,
    real_sh_basis_order3,
    regularized_log_determinant,
    spherical_xyz,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/data/sonicom_reference_grid_v1.csv"
Q26 = ROOT / "configs/data/sonicom_sparse_grid_q26_v1.csv"
OLD_NESTED = ROOT / "configs/data/sonicom_nested_sparse_grid_q6_q14_q26_v1.csv"
OUTPUT = ROOT / "configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.csv"
REPORT = ROOT / "configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def geometry(positions: np.ndarray, indices: list[int]) -> dict[str, object]:
    xyz = spherical_xyz(positions)
    selected = xyz[np.asarray(indices, dtype=np.int64)]
    nearest = np.rad2deg(
        np.arccos(np.clip(np.max(xyz @ selected.T, axis=1), -1.0, 1.0))
    )
    design = real_sh_basis_order3(positions[np.asarray(indices, dtype=np.int64)])
    return {
        "direction_count": len(indices),
        "minimum_pairwise_separation_deg": minimum_pairwise_angle_deg(xyz, indices),
        "mean_covering_distance_deg": float(np.mean(nearest)),
        "p95_covering_distance_deg": float(np.percentile(nearest, 95.0)),
        "maximum_covering_distance_deg": float(np.max(nearest)),
        "sh_order3_rank": int(np.linalg.matrix_rank(design)),
        "sh_order3_condition_number": float(np.linalg.cond(design)),
    }


def extend_q26_to_q50(positions: np.ndarray, q26: list[int]) -> list[int]:
    xyz = spherical_xyz(positions)
    _, pair_orbits = build_lateral_reflection_orbits(positions)
    selected = list(q26)
    selected_set = set(selected)
    available = [orbit for orbit in pair_orbits if selected_set.isdisjoint(orbit)]
    for _ in range(12):
        scored: list[tuple[float, float, int, tuple[int, ...]]] = []
        for orbit in available:
            trial = selected + list(orbit)
            scored.append(
                (
                    round(minimum_pairwise_angle_deg(xyz, trial), 12),
                    regularized_log_determinant(positions, trial),
                    -max(orbit),
                    orbit,
                )
            )
        best = max(scored)[-1]
        selected.extend(best)
        available.remove(best)
    result = sorted(selected)
    if len(result) != 50 or len(set(result)) != 50:
        raise AssertionError("Q50 extension is not 50 unique points")
    return result


def main() -> None:
    reference_rows = read_rows(REFERENCE)
    positions = np.asarray(
        [
            [float(row["azimuth_deg"]), float(row["elevation_deg"])]
            for row in reference_rows
        ],
        dtype=np.float64,
    )
    if positions.shape != (793, 2):
        raise ValueError(f"Unexpected SONICOM reference shape {positions.shape}")

    old_rows = read_rows(OLD_NESTED)
    q14 = [
        int(row["source_index_zero_based"])
        for row in old_rows
        if int(row["direction_count"]) == 14
    ]
    q26 = [int(row["source_index_zero_based"]) for row in read_rows(Q26)]
    q50 = extend_q26_to_q50(positions, q26)
    if not set(q14).issubset(q26) or not set(q26).issubset(q50):
        raise AssertionError("Generated grids are not nested")

    policies = {
        14: "frozen_nested_q14_from_q6_q14_q26_v1",
        26: "frozen_sonicom_q26_v1",
        50: "greedy_lateral_pair_maximin_extension_from_frozen_q26",
    }
    output_rows: list[dict[str, object]] = []
    for count, indices in ((14, q14), (26, q26), (50, q50)):
        for within_index, source_index in enumerate(indices):
            reference = reference_rows[source_index]
            output_rows.append(
                {
                    "sparsity_level": f"Q{count}",
                    "direction_count": count,
                    "within_level_index_zero_based": within_index,
                    "source_index_zero_based": source_index,
                    "azimuth_deg": f"{float(reference['azimuth_deg']):.10f}",
                    "elevation_deg": f"{float(reference['elevation_deg']):.10f}",
                    "colatitude_deg": f"{float(reference['colatitude_deg']):.10f}",
                    "selection_policy": policies[count],
                }
            )
    fields = tuple(output_rows[0].keys())
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "schema_version": "1.0",
        "status": "frozen_geometry",
        "direction_counts": [14, 26, 50],
        "nesting": "Q14 subset Q26 subset Q50",
        "selection_policy": policies,
        "source_indices_zero_based": {"Q14": q14, "Q26": q26, "Q50": q50},
        "geometry": {
            "Q14": geometry(positions, q14),
            "Q26": geometry(positions, q26),
            "Q50": geometry(positions, q50),
        },
        "fixed_evaluation_policy": "Exclude all Q50 inputs at every level",
        "fixed_evaluation_direction_count": 793 - 50,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT).as_posix())
    print(REPORT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
