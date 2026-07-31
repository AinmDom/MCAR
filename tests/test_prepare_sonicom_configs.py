"""Offline tests for SONICOM geometry and split preparation."""

from __future__ import annotations

import numpy as np

from mcar.data_tools.prepare_sonicom_configs import (
    SPLIT_TARGETS,
    allocate_stratum_counts,
    build_lateral_reflection_orbits,
    select_sonicom_q26,
    solid_angle_weights,
)


def synthetic_sonicom_grid() -> np.ndarray:
    rows = []
    for elevation in (
        -45.0,
        -30.0,
        -20.0,
        -10.0,
        0.0,
        10.0,
        20.0,
        30.0,
        45.0,
        60.0,
        75.0,
    ):
        rows.extend(
            (float(azimuth), elevation)
            for azimuth in range(0, 360, 5)
        )
    rows.append((0.0, 90.0))
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    positions = synthetic_sonicom_grid()
    assert positions.shape == (793, 2)
    singletons, pairs = build_lateral_reflection_orbits(positions)
    assert sum(map(len, singletons)) + sum(map(len, pairs)) == 793

    selection = select_sonicom_q26(positions)
    assert len(selection.source_indices) == 26
    assert len(set(selection.source_indices)) == 26
    assert selection.sh_order3_rank == 16
    assert selection.sh_order3_condition_number < 3.0
    assert selection.minimum_pairwise_angle_deg > 30.0
    assert selection.covering_radius_deg < 35.0

    selected_positions = positions[list(selection.source_indices)]
    selected = {
        (round(float(azimuth), 8), round(float(elevation), 8))
        for azimuth, elevation in selected_positions
    }
    for azimuth, elevation in selected:
        assert (
            round(float((-azimuth) % 360.0), 8),
            elevation,
        ) in selected

    weights, domain_area = solid_angle_weights(positions)
    assert weights.shape == (793,)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 1.0)
    expected_cap_area = 2.0 * np.pi * (
        1.0 - np.sin(np.deg2rad(-45.0))
    )
    assert np.isclose(domain_area, expected_cap_area)

    allocation = allocate_stratum_counts(
        {
            "reference_eq_001": 208,
            "reference_eq_002": 19,
            "reference_eq_003": 20,
            "reference_eq_005": 2,
            "reference_eq_006": 32,
            "reference_eq_007": 8,
            "reference_eq_008": 22,
            "reference_eq_009": 6,
            "reference_eq_010": 14,
            "reference_eq_011": 13,
            "reference_eq_012": 6,
        },
        SPLIT_TARGETS,
    )
    for split, expected in SPLIT_TARGETS.items():
        assert sum(row[split] for row in allocation.values()) == expected
    assert all(
        sum(allocation[stratum].values()) == size
        for stratum, size in {
            "reference_eq_001": 208,
            "reference_eq_002": 19,
            "reference_eq_003": 20,
            "reference_eq_005": 2,
            "reference_eq_006": 32,
            "reference_eq_007": 8,
            "reference_eq_008": 22,
            "reference_eq_009": 6,
            "reference_eq_010": 14,
            "reference_eq_011": 13,
            "reference_eq_012": 6,
        }.items()
    )
    print(
        {
            "status": "passed",
            "q26_condition": selection.sh_order3_condition_number,
            "q26_minimum_angle_deg": (
                selection.minimum_pairwise_angle_deg
            ),
            "q26_covering_radius_deg": selection.covering_radius_deg,
            "split_targets": SPLIT_TARGETS,
        }
    )


if __name__ == "__main__":
    main()
